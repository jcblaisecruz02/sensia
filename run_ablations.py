"""Run the sense ablation suite and write metrics/artifacts into --out_dir."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils.inspector import (
    sense_topology_table, plot_rho_histogram,
    procrustes_analysis,
    estimate_jsd_mi_gap,
    index_and_align, collect_embeddings_for_types_cached, plot_aligned_types,
)



def setup_logging(out_dir: Path, level=logging.INFO):
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt_console = "%(message)s"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt_console))

    fmt_file = "[%(asctime)s][%(levelname)s] %(name)s: %(message)s"
    file_handler = logging.FileHandler(out_dir / "ablation.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt_file))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def load_flores(split: str = "dev", src: str = "en", tgt: str = "es"):
    """
    Try to load FLORES via HuggingFace Datasets.
    Fallback: look for parallel text files named flores_{split}.{lang}.txt in CWD.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/flores", f"{src}-{tgt}", split=split)
        return ds
    except Exception:
        src_path = Path(f"flores_{split}.{src}.txt")
        tgt_path = Path(f"flores_{split}.{tgt}.txt")
        if not (src_path.exists() and tgt_path.exists()):
            raise RuntimeError(
                "Could not load FLORES. Install `datasets` or provide parallel files "
                f"{src_path.name} and {tgt_path.name} in the working directory."
            )
        sents_src = src_path.read_text(encoding="utf-8").splitlines()
        sents_tgt = tgt_path.read_text(encoding="utf-8").splitlines()
        assert len(sents_src) == len(sents_tgt), "Parallel files must have same length."
        return [{"sentence_"+src: a, "sentence_"+tgt: b} for a, b in zip(sents_src, sents_tgt)]


def build_aligner(device: str = "cpu"):
    try:
        from simalign import SentenceAligner
        # For speed you may switch to matching_methods="inter".
        return SentenceAligner(model="bert", token_type="bpe", matching_methods="m", device=device)
    except Exception as e:
        raise RuntimeError("SimAlign is required for ablations. Install with `pip install simalign`.") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or HF id of the (LMHead) model")
    ap.add_argument("--tokenizer", required=False, help="Path or HF id of tokenizer (defaults to --model)")
    ap.add_argument("--src", required=True, help="Source language code (e.g., en)")
    ap.add_argument("--tgt", required=True, help="Target language code (e.g., es)")
    ap.add_argument("--split", default="dev", choices=["dev", "devtest", "test"], help="FLORES split")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_pairs", type=int, default=None, help="Cap for aligned word pairs when collecting anchors")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for anchor collection")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if not set)")
    ap.add_argument("--sense_pool_temp", type=float, default=1.0, help="Temperature for sense-mixture pooling")
    ap.add_argument("--viz_types_k", type=int, default=0, help="(Deprecated by --do_sense_visualizations) kept for back-compat; ignored unless flag is on")
    # JSD configuration
    ap.add_argument("--jsd_k", type=int, default=128)
    ap.add_argument("--jsd_pca_dim", type=int, default=32)
    ap.add_argument("--jsd_n_shuffles", type=int, default=10)
    ap.add_argument("--jsd_seed", type=int, default=0)
    # Optional modules
    ap.add_argument("--do_sense_topology", action="store_true", help="Run (1) Sense Topology Correlation")
    ap.add_argument("--do_protocrustes",  action="store_true", help="Run (5) Procrustes / Linear Reconstruction (alias spelling)")
    ap.add_argument("--do_procrustes",    action="store_true", help="Run (5) Procrustes / Linear Reconstruction")
    ap.add_argument("--do_mutual_information", action="store_true", help="Run (7) Mutual Information (JSD drop)")
    ap.add_argument("--do_sense_visualizations", action="store_true", help="Run noun-type sense/context visualizations")
    # Sense topology batching (silently ignored by older inspector versions)
    ap.add_argument("--sense_topology_batch_size", type=int, default=32, help="Sentence batch size for sense topology (if supported)")

    args = ap.parse_args()

    # Run all analyses if no flag is provided (leave visualizations opt-in)
    if not any([args.do_sense_topology, args.do_procrustes, args.do_protocrustes,
                args.do_mutual_information, args.do_sense_visualizations]):
        args.do_sense_topology = True
        args.do_procrustes = True
        args.do_mutual_information = True
        # args.do_sense_visualizations = True  # leave off by default

    if args.do_protocrustes:
        args.do_procrustes = True

    out_dir = Path(args.out_dir)
    setup_logging(out_dir)
    log = logging.getLogger("run_ablations")

    t0 = time.time()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok_id = args.tokenizer or args.model

    log.info(f"Loading model/tokenizer (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True, add_prefix_space=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, trust_remote_code=True)
    model.to(device)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Loading FLORES split={args.split} src={args.src} tgt={args.tgt}")
    ds = load_flores(args.split, args.src, args.tgt)
    log.info("Building aligner (SimAlign)")
    aligner = build_aligner(device=device)

    metrics = {}
    artifacts = {}
    rho_stats = None
    proc_metrics = None
    T_mat = E_mat = None  # anchors cache for MI

    if args.do_sense_topology:
        t1 = time.time()
        log.info("(1) Sense Topology Correlation → computing per-word ρ")
        try:
            df_rho = sense_topology_table(
                model, tokenizer, ds, args.src, args.tgt, aligner,
                max_pairs=args.max_pairs, device=device,
                batch_size_sentences=args.sense_topology_batch_size,
            )
        except TypeError:
            df_rho = sense_topology_table(
                model, tokenizer, ds, args.src, args.tgt, aligner,
                max_pairs=args.max_pairs, device=device,
            )

        plot_rho_histogram(
            df_rho,
            savepath=str(out_dir / "rho_hist.png"),
            title=f"Sense topology ρ ({args.src}↔{args.tgt})",
        )
        rho_stats = {
            "count": int(len(df_rho)),
            "mean": float(df_rho["rho"].mean()) if len(df_rho) else None,
            "median": float(df_rho["rho"].median()) if len(df_rho) else None,
            "std": float(df_rho["rho"].std()) if len(df_rho) else None,
            "min": float(df_rho["rho"].min()) if len(df_rho) else None,
            "max": float(df_rho["rho"].max()) if len(df_rho) else None,
        }
        df_rho.to_csv(out_dir / "sense_topology_rho.csv", index=False)
        log.info(
            f"ρ stats: count={rho_stats['count']} "
            f"mean={rho_stats['mean']:.4f} median={rho_stats['median']:.4f} "
            f"std={rho_stats['std']:.4f}"
        )
        log.info(
            f"Saved: {out_dir/'rho_hist.png'}, {out_dir/'sense_topology_rho.csv'} "
            f"(elapsed {time.time()-t1:.1f}s)"
        )

    if args.do_procrustes:
        t2 = time.time()
        log.info("(5) Procrustes → collecting anchors and fitting Kabsch")
        Q, E_hat, proc_metrics, (T_mat, E_mat) = procrustes_analysis(
            model, tokenizer, ds, args.src, args.tgt, aligner,
            batch_size=args.batch_size, max_pairs=args.max_pairs,
            device=device, weight_mode="mixture", sense_pool_temp=args.sense_pool_temp
        )
        log.info(
            f"Procrustes metrics: mse={proc_metrics['procrustes_mse']:.6f} "
            f"cos={proc_metrics['procrustes_cos']:.6f} "
            f"(elapsed {time.time()-t2:.1f}s)"
        )

    if args.do_mutual_information:
        if T_mat is None or E_mat is None:
            log.info("Collecting anchors for MI (Procrustes not run)")
            _, _, _, (T_mat, E_mat) = procrustes_analysis(
                model, tokenizer, ds, args.src, args.tgt, aligner,
                batch_size=args.batch_size, max_pairs=args.max_pairs,
                device=device, weight_mode="mixture", sense_pool_temp=args.sense_pool_temp
            )

        t3 = time.time()
        log.info("(7) JSD critic (aligned vs. shuffled) → training and evaluating")

        mi_metrics, mi_hist = estimate_jsd_mi_gap(
            E_mat, T_mat,
            steps=800, batch=2048, lr=1e-4, hidden=128,
            eval_shuffles=5, device=device,
        )

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3.5))
            plt.plot(mi_hist)
            plt.xlabel("Step"); plt.ylabel("JSD loss (lower is better)")
            plt.title("JSD Critic Training")
            plt.tight_layout()
            out_png = out_dir / "mi_curve.png"
            plt.savefig(out_png, dpi=200)
            plt.close()
            artifacts["mi_curve"] = str(out_png)
            log.info(f"Saved: {out_png} (elapsed {time.time()-t3:.1f}s)")
        except Exception as e:
            log.warning(f"Could not save MI curve figure: {e}")

        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5.5, 3.6))
            plt.bar(["Aligned", "Shuffled"], [mi_metrics["aligned_mean"], mi_metrics["shuffled_mean"]])
            plt.ylabel("Critic mean (JSD proxy)")
            plt.title(f"ΔMI = {mi_metrics['delta_mi']:.3f}")
            plt.tight_layout()
            drop_png = out_dir / "jsd_delta.png"
            plt.savefig(drop_png, dpi=200)
            plt.close()
            artifacts["jsd_delta"] = str(drop_png)
            log.info(f"Saved: {drop_png}")
        except Exception as e:
            log.warning(f"Could not save JSD ΔMI figure: {e}")

        metrics.setdefault("mutual_information", {}).update(mi_metrics)


    if args.do_sense_visualizations:
        K_vis = args.viz_types_k if args.viz_types_k > 0 else 50
        log.info(f"Noun-type visualizations → top-K types K={K_vis}")
        cached = index_and_align(ds, args.src, args.tgt, aligner)
        from collections import Counter
        cnt = Counter()
        for sen, stg, pairs in cached:
            ws = sen.split(); wt = stg.split()
            for i, j in pairs:
                if i < len(ws) and j < len(wt):
                    a = ws[i].lower(); b = wt[j].lower()
                    if len(a) > 1 and len(b) > 1 and a.isalpha() and b.isalpha():
                        cnt[(a, b)] += 1
        top_types = [ab for ab, _ in cnt.most_common(K_vis)]

        collected, D = collect_embeddings_for_types_cached(
            model, tokenizer, cached, top_types, device=device,
            max_per_type=60, sense_pool_temp=args.sense_pool_temp
        )
        sense_path = out_dir / "nouns_sense.png"
        ctx_path = out_dir / "nouns_ctx.png"
        plot_aligned_types(
            collected, view="sense",
            title=f"Noun alignment (sense) {args.src}↔{args.tgt}",
            savepath=str(sense_path)
        )
        plot_aligned_types(
            collected, view="ctx",
            title=f"Noun alignment (context) {args.src}↔{args.tgt}",
            savepath=str(ctx_path)
        )
        artifacts["nouns_sense"] = str(sense_path)
        artifacts["nouns_ctx"] = str(ctx_path)
        log.info(f"Saved: {sense_path}, {ctx_path}")

    if rho_stats is not None:
        metrics["topology"] = rho_stats
        artifacts["rho_hist"] = str(out_dir / "rho_hist.png")
    if proc_metrics is not None:
        metrics["procrustes"] = proc_metrics
    if T_mat is not None:
        metrics.setdefault("counts", {})["pairs_collected"] = int(T_mat.shape[0])

    (out_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "artifacts": artifacts}, indent=2))
    log.info(f"Wrote metrics: {out_dir/'metrics.json'}")

    log.info(f"Done. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
