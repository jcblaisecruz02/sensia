"""Evaluate FLORES perplexity under different Backpack sense mixing strategies."""
import math
import argparse
import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from typing import Optional, Tuple

LOGGER = logging.getLogger("eval_flores_backpack")

def setup_logging(level: str):
    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_num,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def _get_lm_head(model: nn.Module) -> nn.Module:
    """Return a callable that maps hidden states -> logits (lm_head or output embeddings)."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    emb = model.get_output_embeddings()
    if emb is None:
        raise ValueError("Model has no lm_head or output embeddings.")
    class _Head(nn.Module):
        def __init__(self, emb): 
            super().__init__(); self.emb = emb
        def forward(self, x): 
            return F.linear(x, self.emb.weight, getattr(self.emb, "bias", None))
    return _Head(emb)

def _safe_entropy(p: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> float:
    """Mean entropy H over non-pad tokens; p is [B,K,T]."""
    eps = 1e-12
    H = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1)  # [B,T]
    if token_mask is not None:
        denom = token_mask.sum().clamp_min(1)
        return float((H * token_mask).sum().div(denom).item())
    return float(H.mean().item())

def _infer_mixture(
    fwd_out, per_sense: torch.Tensor, mixture_source: str, tau: float, topk: int,
    attn_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (p_k, topk_mask):
      p_k: [B,K,T] mixture probs for stats / masking
      topk_mask: [B,K,T] one-hot over chosen senses if we enforced top-k here, else None
    Tries model-provided sense weights/logits; falls back to contribution norms.
    """
    B, K, T, D = per_sense.shape
    # Prefer true model outputs if present
    if mixture_source in ("auto", "weights") and hasattr(fwd_out, "sense_weights"):
        p_k = fwd_out.sense_weights  # [B,K,T]
    elif mixture_source in ("auto", "logits") and hasattr(fwd_out, "sense_logits"):
        logits = fwd_out.sense_logits  # [B,K,T]
        if tau != 1.0:
            logits = logits / tau
        # If we want to enforce top-k at the logits level
        if topk and 0 < topk < K:
            topv, topi = torch.topk(logits, topk, dim=1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, topi, 0.0)
            logits = logits + mask
        p_k = torch.softmax(logits, dim=1)
    else:
        # Contribution proxy: ||per_sense|| over D
        contrib = per_sense.norm(dim=-1)  # [B,K,T]
        denom = contrib.sum(dim=1, keepdim=True).clamp_min(1e-9)
        p_k = contrib / denom

    # Top-k on p_k if we didn't already do it on logits
    topk_mask = None
    if topk and 0 < topk < K and not (hasattr(fwd_out, "sense_logits") and mixture_source in ("auto","logits")):
        topi = p_k.topk(topk, dim=1).indices  # [B,topk,T]
        topk_mask = torch.zeros_like(p_k).scatter_(1, topi, 1.0)
        topk_mask = topk_mask / (topk_mask.sum(dim=1, keepdim=True) + 1e-9)
        p_k = (p_k * topk_mask)
        p_k = p_k / (p_k.sum(dim=1, keepdim=True) + 1e-9)

    # Zero out pads in stats (optional)
    if attn_mask is not None and attn_mask.dim() == 2:
        p_k = p_k * attn_mask.unsqueeze(1)
    return p_k, topk_mask

@torch.no_grad()
def ce_from_logits(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).item()

@torch.no_grad()
def mean_offdiag_cos(senses):
    # senses: [B, K, T, D]
    B, K, T, D = senses.shape
    if K <= 1:
        return 0.0
    E = F.normalize(senses.permute(0,2,1,3).reshape(B*T, K, D), dim=-1)
    G = torch.matmul(E, E.transpose(1, 2))  # [B*T, K, K]
    diag_sum = torch.diagonal(G, dim1=1, dim2=2).sum(dim=1)
    off_sum  = G.sum(dim=(1, 2)) - diag_sum
    denom = K * (K - 1)
    return (off_sum / denom).mean().item()

def main():
    ap = argparse.ArgumentParser(description="Evaluate Backpack senses on FLORES (CE full/top1/uniform + optional stats)")
    ap.add_argument("--ckpt", required=True, help="HF repo id or local path to checkpoint")
    ap.add_argument("--lang", default="swh_Latn", help="FLORES-200 language code (e.g. swh_Latn, ind_Latn, est_Latn, tur_Latn)")
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ignore-index", type=int, default=-100)
    ap.add_argument("--compute-entropy", action="store_true")
    ap.add_argument("--compute-offcos", action="store_true")
    ap.add_argument("--mixture-source", default="auto", choices=["auto","contrib","weights","logits"],
                help="Where to derive mixture for entropy/top-k. Prefers model outputs if available.")
    ap.add_argument("--inference-tau", type=float, default=1.0,
                    help="Temperature to apply to sense logits before mixing (if available). 1.0 = no change.")
    ap.add_argument("--inference-topk", type=int, default=1,
                    help="Keep exactly top-k senses per token when forming the 'top-k' variant. Use 1 for classic top-1.")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument("--log-interval", type=int, default=0, help="If >0, log running means every N batches")
    args = ap.parse_args()

    setup_logging(args.log_level)
    device = torch.device(args.device)

    LOGGER.info(f"Loading model from '{args.ckpt}' (trust_remote_code={args.trust_remote_code})")
    config = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=args.trust_remote_code)
    model  = AutoModelForCausalLM.from_pretrained(
        args.ckpt, config=config, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        LOGGER.debug("Tokenizer had no pad_token; set to eos_token.")
    tok.padding_side = "left"

    if not hasattr(model, "backpack"):
        raise AttributeError("Loaded model has no `.backpack` attribute. This script expects your Backpack implementation.")
    backpack = model.backpack

    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Model exposes neither `.lm_head` nor output embeddings.")

    LOGGER.info(f"Loading FLORES-200 '{args.lang}/{args.split}'")
    ds = load_dataset("facebook/flores", args.lang, split=args.split)
    n_total = len(ds)
    LOGGER.info(f"Dataset size: {n_total} sentences")

    tot_full = tot_topk = tot_uniform = 0.0
    entropies, offcos_vals = [], []
    num_batches = 0
    last_K = None

    B = args.batch_size
    pbar = tqdm(range(0, n_total, B), desc="Evaluating", dynamic_ncols=True)

    for i in pbar:
        batch = ds[i:i+B]  # this is a dict of columns
        texts = batch["sentence"]  # list[str]

        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            labels = input_ids.clone()
            if attn_mask is not None and tok.pad_token_id is not None:
                labels = labels.masked_fill(attn_mask == 0, args.ignore_index)
            position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)

            out = backpack(input_ids=input_ids, position_ids=position_ids, labels=labels, attention_mask=enc.get("attention_mask", None), use_cache=False)
            senses = out.senses                  # [B, K, T, D]
            ctxt   = out.contextualization       # [B, K, T, T]
            per_sense = torch.matmul(ctxt, senses)   # [B, K, T, D]

            # Full (sum over senses)
            out_forward = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
            full_logits = out_forward.logits
            ce_full = ce_from_logits(full_logits, labels, ignore_index=args.ignore_index)

            # Pseudo mixture p_k ∝ ||contribution|| for stats/ablations
            contrib = per_sense.norm(dim=-1)  # [B, K, T]
            denom = (contrib.sum(dim=1, keepdim=True) + 1e-9)
            p_k = contrib / denom

            if args.compute_entropy:
                H = -(p_k.clamp_min(1e-9) * p_k.clamp_min(1e-9).log()).sum(dim=1)  # [B, T]
                entropies.append(H.mean().item())

            if args.compute_offcos:
                offcos_vals.append(mean_offdiag_cos(senses))

            # Top-k
            # Get mixture p_k and optional topk mask
            p_k, mask_topk = _infer_mixture(
                out, per_sense, args.mixture_source, args.inference_tau, args.inference_topk, attn_mask
            )

            # If we didn't enforce top-k inside _infer_mixture, build it now from p_k
            topk_mask = mask_topk
            if topk_mask is None:
                # number of senses (K) lives on dim=1 of p_k: [B, K, T]
                K = p_k.size(1)
                # normalize/clip k to [1, K]
                k = int(args.inference_topk) if args.inference_topk and args.inference_topk > 0 else 1
                k = min(k, K)

                # pick top-k senses per token along the sense dim
                topi = p_k.topk(k, dim=1).indices  # [B, k, T]
                topk_mask = torch.zeros_like(p_k).scatter_(1, topi, 1.0)  # [B, K, T]
                # renormalize over the selected senses so weights sum to 1
                topk_mask = topk_mask / (topk_mask.sum(dim=1, keepdim=True) + 1e-9)


            # Top-k hidden → logits → CE
            topk_hidden = (per_sense * topk_mask.unsqueeze(-1)).sum(dim=1)  # [B,T,D]
            topk_logits = _get_lm_head(model)(topk_hidden)
            ce_topk = ce_from_logits(topk_logits, labels, ignore_index=args.ignore_index)

            # Uniform
            uniform_hidden = per_sense.mean(dim=1)  # [B,T,D]
            uniform_logits = _get_lm_head(model)(uniform_hidden)
            ce_uniform = ce_from_logits(uniform_logits, labels, ignore_index=args.ignore_index)

            last_K = p_k.size(1)

        # Aggregate
        tot_full   += ce_full
        tot_topk   += ce_topk
        tot_uniform+= ce_uniform
        num_batches += 1

        # Update tqdm postfix
        pbar.set_postfix({
            "CE_full": f"{tot_full/num_batches:.3f}",
            "CE_topk": f"{tot_topk/num_batches:.3f}",
            "CE_unif": f"{tot_uniform/num_batches:.3f}",
        })

        # Optional periodic logging
        if args.log_interval and (num_batches % args.log_interval == 0):
            LOGGER.info(
                f"[{num_batches} / {math.ceil(n_total/B)}] "
                f"CE(full)={tot_full/num_batches:.3f}  "
                f"CE(topk)={tot_topk/num_batches:.3f}  "
                f"CE(unif)={tot_uniform/num_batches:.3f}"
            )

    # Final means
    mean_ce_full    = tot_full / num_batches
    mean_ce_topk    = tot_topk / num_batches
    mean_ce_uniform = tot_uniform / num_batches

    LOGGER.info("=== FLORES Full-Split Summary ===")
    LOGGER.info(f"Mean CE(full)      = {mean_ce_full:.3f}")
    LOGGER.info(f"Mean CE(top-k)     = {mean_ce_topk:.3f}")
    LOGGER.info(f"Mean CE(uniform)   = {mean_ce_uniform:.3f}")

    if args.compute_entropy:
        mean_entropy = sum(entropies)/len(entropies) if entropies else float('nan')
        logK = math.log(last_K) if last_K else float('nan')
        LOGGER.info(f"Usage entropy mean = {mean_entropy:.3f} (log K = {logK:.3f})")

    if args.compute_offcos:
        mean_offcos = sum(offcos_vals)/len(offcos_vals) if offcos_vals else float('nan')
        LOGGER.info(f"Off-diag cosine    = {mean_offcos:.3f}")

if __name__ == "__main__":
    main()
