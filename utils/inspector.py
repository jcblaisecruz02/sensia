from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"^[\\W_]+$", re.UNICODE)


def _require_fast_tokenizer(tokenizer):
    """Raise a RuntimeError if the provided tokenizer is not a fast tokenizer."""
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("This requires a *fast* tokenizer (supports .word_ids()).")


def _center(x: torch.Tensor) -> torch.Tensor:
    """Remove the batch mean from a tensor along dim 0."""
    return x - x.mean(dim=0, keepdim=True)


def _get_ctx(outputs, i: Optional[int] = None) -> Optional[torch.Tensor]:
    """Return last-layer hidden states for item i (or the whole batch)."""
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return None
    hs = outputs.hidden_states[-1]
    return hs if i is None else hs[i]


def _word_to_subword_spans_fast(tokenizer, words: List[str]):
    """Map word indices -> subword span positions for a single sentence (fast tokenizer)."""
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    wi = enc.word_ids(0)  # list[int|None]
    spans: Dict[int, List[int]] = {}
    for pos, widx in enumerate(wi):
        if widx is None:
            continue
        spans.setdefault(widx, []).append(pos)
    return spans, enc


def _batch_word_spans_fast(tokenizer, batch_words: List[List[str]]):
    """Batched word->subword spans; returns (encoding, list_of_spans)."""
    enc = tokenizer(
        batch_words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    spans_list: List[Dict[int, List[int]]] = []
    for i in range(len(batch_words)):
        wi = enc.word_ids(i)  # list[int|None]
        spans: Dict[int, List[int]] = {}
        for pos, widx in enumerate(wi):
            if widx is None:
                continue
            spans.setdefault(widx, []).append(pos)
        spans_list.append(spans)
    return enc, spans_list


def _sense_weights_from_context(S_k_p_d: torch.Tensor,
                                H_p_d: Optional[torch.Tensor],
                                temp: float = 1.0) -> torch.Tensor:
    """
    Compute mixture weights over K senses, guided by context span H (mean pooled).
    S_k_p_d: [K, P, D], H_p_d: [P, D] or None -> returns [K] softmax weights.
    """
    K = S_k_p_d.size(0)
    if H_p_d is None:
        return torch.full((K,), 1.0 / K, device=S_k_p_d.device)
    scores = torch.einsum("pd,kpd->k", H_p_d, S_k_p_d) / max(1e-8, float(temp))
    return torch.softmax(scores, dim=0)


def _is_content_token(w: Optional[str]) -> bool:
    """Heuristically detect if a token should count as content (non-punctuation, len>1)."""
    return (w is not None) and (len(w) > 1) and (not _PUNCT.match(w))


def _resolve_backpack(model):
    """Return the backpack base (your model has .backpack on LMHead)"""
    return getattr(model, "backpack", model)


# ------------------------------
# (1) Sense topology correlation
# ------------------------------

@torch.no_grad()
def _spearman_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman’s ρ as Pearson of ranks (torch-only; approx ties)."""
    def ranks(v: torch.Tensor):
        idx = torch.argsort(v)
        r = torch.empty_like(v, dtype=torch.float)
        r[idx] = torch.arange(1, v.numel()+1, device=v.device, dtype=torch.float)
        return r
    rx = ranks(x); ry = ranks(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = torch.sqrt((rx**2).sum() * (ry**2).sum())
    if denom.item() == 0:
        return 0.0
    return float((rx @ ry) / denom)

@torch.no_grad()
def sense_topology_table(model,
                         tokenizer,
                         ds: Iterable[Dict[str, Any]],
                         src: str,
                         tgt: str,
                         aligner,
                         max_pairs: Optional[int] = None,
                         device: Optional[str] = None,
                         batch_size_sentences: int = 32) -> pd.DataFrame:
    """
    FAST version:
      - Batches sentence forwards (GPU efficient)
      - Precomputes per-word sense-set Gram upper-triangle vectors once
      - Uses torch-only Spearman to avoid SciPy/numpy overhead per pair
    Returns DataFrame: [sentence_idx, word_src, word_tgt, rho]
    """
    device = device or (model.device if hasattr(model, "device") else "cpu")
    model.eval().to(device)
    _require_fast_tokenizer(tokenizer)

    rows, it = [], 0
    try:
        total = len(ds)
        data_list = ds
    except Exception:
        data_list = list(ds)
        total = len(data_list)

    from tqdm.auto import tqdm
    pbar = tqdm(range(0, total, batch_size_sentences),
                desc="(1) Sense Topology: sentences (batched)", leave=False)
    for start in pbar:
        end = min(start + batch_size_sentences, total)
        batch = [data_list[i] for i in range(start, end)]

        sents_src, sents_tgt = [], []
        words_src_all, words_tgt_all, pairs_all = [], [], []
        for ex in batch:
            sen = ex.get(f"sentence_{src}"); stg = ex.get(f"sentence_{tgt}")
            if not sen or not stg:
                sents_src.append(None); sents_tgt.append(None)
                words_src_all.append([]); words_tgt_all.append([]); pairs_all.append([])
                continue
            ws = sen.split(); wt = stg.split()
            alns = aligner.get_word_aligns(sen, stg)
            pairs = alns.get("mwmf", []) or alns.get("m", []) or alns.get("inter", [])
            pairs = [(i, j) for (i, j) in pairs
                     if i < len(ws) and j < len(wt)
                     and _is_content_token(ws[i]) and _is_content_token(wt[j])]
            sents_src.append(sen); sents_tgt.append(stg)
            words_src_all.append(ws); words_tgt_all.append(wt); pairs_all.append(pairs)

        valid_idx = [i for i,(a,b) in enumerate(zip(sents_src, sents_tgt)) if (a and b and pairs_all[i])]
        if not valid_idx:
            continue

        enc_src, spans_src_list = _batch_word_spans_fast(tokenizer, [words_src_all[i] for i in valid_idx])
        enc_tgt, spans_tgt_list = _batch_word_spans_fast(tokenizer, [words_tgt_all[i] for i in valid_idx])
        enc_src = {k: v.to(device) for k, v in enc_src.items()}
        enc_tgt = {k: v.to(device) for k, v in enc_tgt.items()}

        out_src = model(**enc_src, return_dict=True)
        out_tgt = model(**enc_tgt, return_dict=True)
        S_src = out_src.senses  # [B,K,S,D]
        S_tgt = out_tgt.senses  # [B,K,S,D]

        K = S_src.size(1)
        iu = torch.triu_indices(K, K, offset=1)

        per_sent_word_gram_src, per_sent_word_gram_tgt = [], []
        for local_i, g_i in enumerate(valid_idx):
            spans_src = spans_src_list[local_i]; spans_tgt = spans_tgt_list[local_i]
            Se_i = S_src[local_i]  # [K,S,D]; St_i = S_tgt[local_i]
            St_i = S_tgt[local_i]

            grams_src = {}
            for wi, span in spans_src.items():
                if not span: continue
                vecs = Se_i[:, span, :].mean(dim=1)
                vecs = F.normalize(vecs - vecs.mean(0, keepdim=True), dim=-1)
                G = (vecs @ vecs.t()).clamp(-1,1)
                grams_src[wi] = G[iu[0], iu[1]]

            grams_tgt = {}
            for wj, span in spans_tgt.items():
                if not span: continue
                vecs = St_i[:, span, :].mean(dim=1)
                vecs = F.normalize(vecs - vecs.mean(0, keepdim=True), dim=-1)
                G = (vecs @ vecs.t()).clamp(-1,1)
                grams_tgt[wj] = G[iu[0], iu[1]]

            per_sent_word_gram_src.append(grams_src)
            per_sent_word_gram_tgt.append(grams_tgt)

        for local_i, g_i in enumerate(valid_idx):
            ws = words_src_all[g_i]; wt = words_tgt_all[g_i]
            grams_src = per_sent_word_gram_src[local_i]
            grams_tgt = per_sent_word_gram_tgt[local_i]
            for wi_src, wi_tgt in pairs_all[g_i]:
                if wi_src not in grams_src or wi_tgt not in grams_tgt:
                    continue
                vE = grams_src[wi_src]; vT = grams_tgt[wi_tgt]
                rho = _spearman_torch(vE, vT)
                rows.append({
                    "sentence_idx": start + local_i,
                    "word_src": ws[wi_src],
                    "word_tgt": wt[wi_tgt],
                    "rho": float(rho),
                })
                it += 1
                if max_pairs is not None and it >= max_pairs:
                    logger.info(f"Reached max_pairs={max_pairs} in sense_topology_table (fast).")
                    pbar.close()
                    return pd.DataFrame(rows, columns=["sentence_idx", "word_src", "word_tgt", "rho"])

    return pd.DataFrame(rows, columns=["sentence_idx", "word_src", "word_tgt", "rho"])


# -------------------------------------------
# (5) Linear Reconstruction / Procrustes
# -------------------------------------------

@torch.no_grad()
def procrustes_fit(T: torch.Tensor, E: torch.Tensor):
    """
    Solve for orthogonal Q minimizing ||(T - mean(T)) Q - (E - mean(E))||_F.
    Returns (Q, E_hat, metrics) where E_hat = centered(T) @ Q.

    metrics = {"procrustes_mse": float, "procrustes_cos": float}
    """
    T0, E0 = _center(T), _center(E)
    M = T0.T @ E0
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    E_hat = T0 @ Q
    mse = torch.mean((E_hat - E0) ** 2).item()
    cos = F.cosine_similarity(E_hat, E0, dim=-1).mean().item()
    return Q, E_hat, {"procrustes_mse": mse, "procrustes_cos": cos}


@torch.no_grad()
def collect_pairs_batched(model,
                          tokenizer,
                          ds: Iterable[Dict[str, Any]],
                          src: str,
                          tgt: str,
                          aligner,
                          batch_size: int = 64,
                          max_pairs: Optional[int] = None,
                          device: Optional[str] = None,
                          weight_mode: str = "mixture",
                          sense_pool_temp: float = 1.0,
                          dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (T_mat, E_mat) as anchors for global analysis.
    Each row is a (sense-level or context-level) vector pooled over an aligned word pair.
    """
    device = device or (model.device if hasattr(model, "device") else "cpu")
    model.eval().to(device)
    _require_fast_tokenizer(tokenizer)

    E_rows, T_rows = [], []
    it = 0

    batch_sent_src: List[str] = []
    batch_sent_tgt: List[str] = []
    batch_pairs: List[List[Tuple[int, int]]] = []

    # Try to estimate total pairs for progress
    try:
        total_sent = len(ds)
    except Exception:
        total_sent = None

    pbar = tqdm(ds, total=total_sent, desc="(5) Collecting anchors", leave=False)

    def flush_batch():
        nonlocal E_rows, T_rows, it
        if not batch_sent_src:
            return
        enc_src, spans_src = _batch_word_spans_fast(tokenizer, [s.split() for s in batch_sent_src])
        enc_tgt, spans_tgt = _batch_word_spans_fast(tokenizer, [s.split() for s in batch_sent_tgt])
        enc_src = {k: v.to(device) for k, v in enc_src.items()}
        enc_tgt = {k: v.to(device) for k, v in enc_tgt.items()}

        out_src = model(**enc_src, return_dict=True, output_hidden_states=True)
        out_tgt = model(**enc_tgt, return_dict=True, output_hidden_states=True)

        S_src = out_src.senses  # [B,K,S,D]
        S_tgt = out_tgt.senses  # [B,K,S,D]
        H_src = _get_ctx(out_src)  # [B,S,D] or None
        H_tgt = _get_ctx(out_tgt)

        B = S_src.size(0)
        for i in range(B):
            K = S_src.size(1)
            Se_i = S_src[i]  # [K,S,D]
            St_i = S_tgt[i]  # [K,S,D]
            He_i = H_src[i] if H_src is not None else None
            Ht_i = H_tgt[i] if H_tgt is not None else None

            for wi_src, wi_tgt in batch_pairs[i]:
                pos_src = spans_src[i].get(wi_src)
                pos_tgt = spans_tgt[i].get(wi_tgt)
                if not pos_src or not pos_tgt:
                    continue

                # sense sets
                Se = Se_i[:, pos_src, :]  # [K,P,D]
                St = St_i[:, pos_tgt, :]  # [K,P,D]

                # weights
                if weight_mode == "uniform":
                    we = torch.full((K,), 1.0 / K, device=device)
                    wt = torch.full((K,), 1.0 / K, device=device)
                else:  # "mixture" uses context to gate senses
                    He_span = He_i[pos_src, :] if He_i is not None else None
                    Ht_span = Ht_i[pos_tgt, :] if Ht_i is not None else None
                    we = _sense_weights_from_context(Se, He_span, temp=sense_pool_temp)
                    wt = _sense_weights_from_context(St, Ht_span, temp=sense_pool_temp)

                e_vecs = Se.mean(dim=1)  # [K,D]
                t_vecs = St.mean(dim=1)  # [K,D]
                v_src = (we[:, None] * e_vecs).sum(dim=0)  # [D]
                v_tgt = (wt[:, None] * t_vecs).sum(dim=0)  # [D]

                E_rows.append(v_src.detach().to(device="cpu", dtype=torch.float32))
                T_rows.append(v_tgt.detach().to(device="cpu", dtype=torch.float32))

                it += 1
                if max_pairs is not None and it >= max_pairs:
                    logger.info(f"Reached max_pairs={max_pairs} while collecting anchors.")
                    return

    for ex in pbar:
        sen = ex.get(f"sentence_{src}")
        stg = ex.get(f"sentence_{tgt}")
        if not sen or not stg:
            continue

        alns = aligner.get_word_aligns(sen, stg)
        pairs = alns.get("mwmf", []) or alns.get("m", []) or alns.get("inter", [])
        # filter to content-ish tokens quickly
        words_src = sen.split()
        words_tgt = stg.split()
        filtered = [(i, j) for (i, j) in pairs
                    if i < len(words_src) and j < len(words_tgt)
                    and _is_content_token(words_src[i]) and _is_content_token(words_tgt[j])]
        if not filtered:
            continue

        batch_sent_src.append(sen)
        batch_sent_tgt.append(stg)
        batch_pairs.append(filtered)

        if len(batch_sent_src) >= batch_size:
            flush_batch()
            batch_sent_src.clear(); batch_sent_tgt.clear(); batch_pairs.clear()
            if max_pairs is not None and it >= max_pairs:
                break

    # flush remainder
    if batch_sent_src and (max_pairs is None or it < max_pairs):
        flush_batch()

    pbar.close()

    if not E_rows:
        raise RuntimeError("No anchors collected; check tokenizer fast mode and outputs.senses shape.")

    E_mat = torch.stack(E_rows, 0).float()
    T_mat = torch.stack(T_rows, 0).float()
    return T_mat, E_mat


@torch.no_grad()
def procrustes_analysis(model,
                        tokenizer,
                        ds: Iterable[Dict[str, Any]],
                        src: str,
                        tgt: str,
                        aligner,
                        batch_size: int = 64,
                        max_pairs: Optional[int] = None,
                        device: Optional[str] = None,
                        weight_mode: str = "mixture",
                        sense_pool_temp: float = 1.0,
                        dtype: torch.dtype = torch.bfloat16):
    """
    Collection + Kabsch fit in one call.
    Returns (Q, E_hat, metrics, (T, E))
    """
    T, E = collect_pairs_batched(
        model, tokenizer, ds, src, tgt, aligner,
        batch_size=batch_size, max_pairs=max_pairs, device=device,
        weight_mode=weight_mode, sense_pool_temp=sense_pool_temp, dtype=dtype
    )
    logger.info(f"Collected anchors: {T.shape[0]} pairs, dim={T.shape[1]}")
    Q, E_hat, metrics = procrustes_fit(T, E)
    logger.info(f"Procrustes: mse={metrics['procrustes_mse']:.6f} cos={metrics['procrustes_cos']:.6f}")
    return Q, E_hat, metrics, (T, E)


# -------------------------
# (7) Mutual Information
# -------------------------

class _JSDCritic(nn.Module):
    """Two-layer MLP critic used to contrast aligned vs shuffled pairs for MI estimation."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),   nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # returns logits, shape [B]
        return self.f(torch.cat([x, y], dim=-1)).squeeze(-1)

def _standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean-center and variance-normalize features for stabler critic training."""
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return x / std

@torch.no_grad()
def _mean_score(critic: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                batch: int = 4096, device: Optional[str] = None) -> float:
    """Evaluate average critic output on batched data without tracking gradients."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N = X.size(0)
    s = 0.0
    n = 0
    for i in range(0, N, batch):
        xb = X[i:i+batch].to(device)
        yb = Y[i:i+batch].to(device)
        logits = critic(xb, yb).detach()
        s += logits.sum().item()
        n += logits.numel()
    return s / max(1, n)

def estimate_jsd_mi_gap(
    E: torch.Tensor,
    T: torch.Tensor,
    steps: int = 800,
    batch: int = 2048,
    lr: float   = 1e-4,
    hidden: int = 128,
    eval_shuffles: int = 5,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], List[float]]:
    """
    Train an f-GAN JSD critic Tθ(x,y) on aligned vs shuffled pairs.
    Returns:
      metrics = {
        "aligned_mean": <critic mean on aligned>,
        "shuffled_mean": <mean critic on shuffled>,
        "delta_mi": aligned_mean - shuffled_mean,
      }
      hist = per-step objective values (lower is better).
    """
    assert E.shape == T.shape, "E and T must have same shape"
    N, D = E.shape
    if N < 4:
        return {"aligned_mean": float("nan"), "shuffled_mean": float("nan"), "delta_mi": float("nan")}, []

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Standardize features for stability (doesn't change the dependency signal)
    E = _standardize(E).to(device)
    T = _standardize(T).to(device)

    critic = _JSDCritic(D, hidden=hidden).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)

    hist: List[float] = []
    critic.train()
    for _ in range(steps):
        bsz = min(batch, N)
        idx = torch.randint(0, N, (bsz,), device=device)
        x = E[idx]
        y = T[idx]
        y_shuf = y[torch.randperm(bsz, device=device)]

        pos = critic(x, y).clamp(-50, 50)       # aligned logits
        neg = critic(x, y_shuf).clamp(-50, 50)  # shuffled logits

        # f-GAN JSD (maximize) => minimize softplus(-pos) + softplus(neg)
        loss = F.softplus(-pos).mean() + F.softplus(neg).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()

        hist.append(float(loss.detach().cpu().item()))

    critic.eval()
    aligned_mean = _mean_score(critic, E.cpu(), T.cpu(), device=device)

    # average over a few random shuffles
    shuf_scores = []
    for _ in range(eval_shuffles):
        perm = torch.randperm(N)
        shuffled_mean = _mean_score(critic, E.cpu(), T.cpu()[perm], device=device)
        shuf_scores.append(shuffled_mean)
    shuffled_mean = float(np.mean(shuf_scores))

    metrics = {
        "aligned_mean": float(aligned_mean),
        "shuffled_mean": float(shuffled_mean),
        "delta_mi": float(aligned_mean - shuffled_mean),
        "steps": int(steps),
        "batch": int(batch),
        "lr": float(lr),
        "hidden": int(hidden),
        "eval_shuffles": int(eval_shuffles),
        "N_pairs": int(N),
        "dim": int(D),
    }
    return metrics, hist

# ----------------
# Visualization
# ----------------

def plot_rho_histogram(df: pd.DataFrame,
                       savepath: Optional[str] = None,
                       title: str = "Sense topology correlation (ρ)"):
    """Plot a histogram of Spearman correlations, filtering out degenerate entries."""
    import matplotlib.pyplot as plt

    df_plot = df.copy()
    # basic cleaning: drop extreme sentinels; non-alnum
    df_plot = df_plot[(df_plot["rho"] < 0.99) & (df_plot["rho"] > -0.99)]
    df_plot = df_plot[df_plot["word_src"].apply(_is_content_token)]
    df_plot = df_plot[df_plot["word_tgt"].apply(_is_content_token)]

    plt.figure(figsize=(7, 4))
    plt.hist(df_plot["rho"], bins=50, edgecolor="black")
    plt.xlabel("Spearman ρ")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()
    else:
        plt.show()


def index_and_align(ds, src, tgt, aligner):
    """Cache alignments for a dataset split; returns a list of (sen, stg, pairs)."""
    cached = []
    try:
        total = len(ds)
    except Exception:
        total = None

    for ex in tqdm(ds, total=total, desc="Index+align (cache)", leave=False):
        sen = ex.get(f"sentence_{src}"); stg = ex.get(f"sentence_{tgt}")
        if not sen or not stg:
            continue
        alns = aligner.get_word_aligns(sen, stg)
        pairs = alns.get("mwmf", []) or alns.get("m", []) or alns.get("inter", [])
        cached.append((sen, stg, pairs))
    return cached

@torch.no_grad()
def collect_embeddings_for_types_cached(model,
                                        tokenizer,
                                        cached: Sequence[Tuple[str, str, List[Tuple[int, int]]]],
                                        types: Sequence[Tuple[str, str]],
                                        device: Optional[str] = None,
                                        max_per_type: int = 60,
                                        sense_pool_temp: float = 1.0):
    """
    For a set of aligned noun 'types' (surface forms), collect sense- and context-level vectors
    for projection. Returns (rows, D) where rows is a list of dicts with:
      {"word": <str>, "lang": "src"/"tgt", "sense": [D], "ctx": [D]}
    """
    device = device or (model.device if hasattr(model, "device") else "cpu")
    model.eval().to(device)
    _require_fast_tokenizer(tokenizer)

    rows: List[Dict[str, Any]] = []
    D_out: Optional[int] = None
    wanted = set(types)

    from tqdm.auto import tqdm
    pbar = tqdm(cached, desc="Collect noun-type embeddings", leave=False)
    for sen, stg, pairs in pbar:
        words_src = sen.split(); words_tgt = stg.split()

        # Correct order: spans first, then encoding
        spans_src, enc_src = _word_to_subword_spans_fast(tokenizer, words_src)
        spans_tgt, enc_tgt = _word_to_subword_spans_fast(tokenizer, words_tgt)
        enc_src = {k: v.to(device) for k, v in enc_src.items()}
        enc_tgt = {k: v.to(device) for k, v in enc_tgt.items()}

        out_src = model(**enc_src, return_dict=True, output_hidden_states=True)
        out_tgt = model(**enc_tgt, return_dict=True, output_hidden_states=True)

        S_src = out_src.senses[0]  # [K,S,D]
        S_tgt = out_tgt.senses[0]  # [K,S,D]
        H_src = _get_ctx(out_src)  # [S,D] or None
        H_tgt = _get_ctx(out_tgt)

        if D_out is None:
            D_out = int(S_src.size(-1))

        per_type_counts: Dict[Tuple[str, str], int] = {}

        for wi_src, wi_tgt in pairs:
            if wi_src >= len(words_src) or wi_tgt >= len(words_tgt):
                continue
            key = (words_src[wi_src].lower(), words_tgt[wi_tgt].lower())
            if key not in wanted:
                continue
            if per_type_counts.get(key, 0) >= max_per_type:
                continue

            pos_src = spans_src.get(wi_src)
            pos_tgt = spans_tgt.get(wi_tgt)
            if not pos_src or not pos_tgt:
                continue

            Se = S_src[:, pos_src, :]  # [K,P,D]
            St = S_tgt[:, pos_tgt, :]  # [K,P,D]

            He = H_src[pos_src, :] if H_src is not None else None
            Ht = H_tgt[pos_tgt, :] if H_tgt is not None else None

            we = _sense_weights_from_context(Se, He, temp=sense_pool_temp)
            wt = _sense_weights_from_context(St, Ht, temp=sense_pool_temp)

            e_vecs = Se.mean(dim=1)  # [K,D]
            t_vecs = St.mean(dim=1)  # [K,D]

            v_src_sense = (we[:, None] * e_vecs).sum(dim=0)
            v_tgt_sense = (wt[:, None] * t_vecs).sum(dim=0)
            v_src_ctx = (He.mean(dim=0) if He is not None else e_vecs.mean(dim=0))
            v_tgt_ctx = (Ht.mean(dim=0) if Ht is not None else t_vecs.mean(dim=0))

            rows.append({"word": key[0], "lang": "src",
                         "sense": v_src_sense.detach().cpu().numpy(),
                         "ctx": v_src_ctx.detach().cpu().numpy()})
            rows.append({"word": key[1], "lang": "tgt",
                         "sense": v_tgt_sense.detach().cpu().numpy(),
                         "ctx": v_tgt_ctx.detach().cpu().numpy()})

            per_type_counts[key] = per_type_counts.get(key, 0) + 1

    pbar.close()
    return rows, int(D_out or 0)

def plot_aligned_types(collected_rows: Sequence[Dict[str, Any]],
                       view: str = "sense",
                       title: Optional[str] = None,
                       savepath: Optional[str] = None,
                       max_points_per_type: int = 200):
    """
    Project aligned noun-type vectors to 2D (PCA) and plot, colored by language,
    with per-type markers and a bottom legend.
    """
    import matplotlib.pyplot as plt

    rows = collected_rows
    words = sorted({r["word"] for r in rows})
    # Build matrices
    X = np.vstack([r[view] for r in rows])  # [N, D]
    # PCA to 2D
    X0 = X - X.mean(0, keepdims=True)
    U, S, VT = np.linalg.svd(X0, full_matrices=False)
    Z = X0 @ VT[:2].T

    # Plot
    plt.figure(figsize=(7.5, 6.0))
    ax = plt.gca()
    # one color per lang, different marker per word
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]
    marker_map = {w: markers[i % len(markers)] for i, w in enumerate(words)}

    # Limit points per type
    per_type = {w: 0 for w in words}
    for r, point in zip(rows, Z):
        w = r["word"]
        if per_type[w] >= max_points_per_type:
            continue
        m = marker_map[w]
        if r["lang"] == "src":
            ax.scatter(point[0], point[1], marker=m, alpha=0.7, label=w, zorder=2)
        else:
            ax.scatter(point[0], point[1], marker=m, alpha=0.7, facecolors="none",
                       edgecolors="black", label=w, zorder=3)
        per_type[w] += 1

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"Aligned types projection ({view}-level)")

    # Legend (deduplicated) at bottom
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_h.append(h); uniq_l.append(l)

    ncols = min(6, len(uniq_l))
    ax.legend(uniq_h, uniq_l, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=ncols, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if savepath:
        plt.savefig(savepath, dpi=200)
        plt.close()
    else:
        plt.show()
