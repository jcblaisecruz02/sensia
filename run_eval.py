#!/usr/bin/env python
"""Evaluate Backpack models on Belebele, XCOPA, XStoryCloze, or Tatoeba."""

import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, VerificationMode

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Language mapping utilities
try:
    from data import FLORES_CODE_MAP as _FLORES
except Exception:
    _FLORES = {}

THREE_TO_TWO = {
    "est": "et", "ind": "id", "swh": "sw", "tur": "tr", "zho": "zh"
}
TWO_TO_THREE = {
    "en": "eng", "tr": "tur", "sw": "swa", "id": "ind", "et": "est", "zh": "zho"
}
FLORES_FALLBACKS = {"est": "est_Latn", "ind": "ind_Latn", "swh": "swh_Latn", "tur": "tur_Latn", "zho": "zho_Hans"}

def map_lang(task: str, lang3: str) -> str:
    l = (lang3 or "").lower()
    if task == "belebele":
        if "_" in l:  # already FLORES
            return l
        if l in _FLORES:
            return _FLORES[l]
        return FLORES_FALLBACKS.get(l, l)
    # xcopa / xstorycloze / tatoeba → 2-letter
    return THREE_TO_TWO.get(l, l)

def to_iso639_3(two_letter: str) -> str:
    if two_letter not in TWO_TO_THREE:
        raise ValueError(f"Unsupported 2-letter code '{two_letter}' for tatoeba_mt. "
                         f"Known: {sorted(TWO_TO_THREE.keys())}")
    return TWO_TO_THREE[two_letter]

# Model and tokenizer loading
def load_model_and_tokenizer(model_path: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"  # for MCQ we keep the tail (answer span)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, trust_remote_code=True).to(device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    return model, tok

# Shared helpers
def _tok_ids(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False)["input_ids"]

def _align_labels(full_ids: List[int], ans_ids: List[int]) -> Tuple[List[int], int]:
    L, m = len(full_ids), len(ans_ids)
    pos = -1
    for s in range(L - m, -1, -1):
        if full_ids[s:s+m] == ans_ids:
            pos = s; break
    labels = [-100]*L
    if pos >= 0:
        labels[pos:pos+m] = full_ids[pos:pos+m]
        return labels, m
    m = min(m, L)
    labels[-m:] = full_ids[-m:]
    return labels, m

def _normalize(norm_len: torch.Tensor, use_mean: bool, alpha: float) -> torch.Tensor:
    if use_mean: return norm_len.clamp_min(1.0)
    if alpha > 0: return norm_len.clamp_min(1.0).pow(alpha)
    return torch.ones_like(norm_len)

@torch.no_grad()
def _sum_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
             device: torch.device, fp16: bool) -> torch.Tensor:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100, reduction="none"
        ).view(shift_labels.size())
        mask = (shift_labels != -100).float()
        return (loss_tok * mask).sum(dim=1)  # [B]

# Multiple-choice task iterators
def iter_belebele(ds, tok, max_len, batch, use_mean, alpha):
    nl = _tok_ids(tok, "\n")
    qpref = _tok_ids(tok, "Q: ")
    apref = _tok_ids(tok, "\nA: ")
    for s in range(0, len(ds), batch):
        e = min(s+batch, len(ds))
        ids_list, labels_list, norm_list, gold = [], [], [], []
        for ex in ds.select(range(s, e)):
            passage = ex.get("flores_passage","").strip()
            question = ex.get("question","").strip()
            answers = [ex.get("mc_answer1","").strip(), ex.get("mc_answer2","").strip(),
                       ex.get("mc_answer3","").strip(), ex.get("mc_answer4","").strip()]
            g = max(0, min(3, int(ex.get("correct_answer_num",1))-1))
            ctx = _tok_ids(tok, passage) + nl + nl + qpref + _tok_ids(tok, question)
            for ans in answers:
                a_ids = _tok_ids(tok, ans)
                tail = apref + a_ids
                budget = max_len - len(tail)
                ctx_cut = ctx[-budget:] if budget>0 and len(ctx)>budget else ctx if budget>0 else []
                full = (ctx_cut + tail)[-max_len:]
                lab, L = _align_labels(full, a_ids)
                ids_list.append(full); labels_list.append(lab); norm_list.append(L)
            gold.append(g)
        enc = tok.pad({"input_ids": ids_list}, padding=True, max_length=max_len, return_tensors="pt")
        Lmax = enc["input_ids"].size(1)
        labels = torch.tensor([lab + [-100]*(Lmax-len(lab)) if len(lab)<Lmax else lab[:Lmax] for lab in labels_list], dtype=torch.long)
        norm = _normalize(torch.tensor(norm_list, dtype=torch.float), use_mean, alpha)
        yield enc["input_ids"], enc["attention_mask"], labels, norm, gold, 4

def iter_xcopa(ds, tok, max_len, batch, use_mean, alpha, template: Optional[str]):
    for s in range(0, len(ds), batch):
        e = min(s+batch, len(ds))
        ids_list, labels_list, norm_list, gold = [], [], [], []
        for ex in ds.select(range(s, e)):
            premise = ex["premise"].strip()
            c1, c2 = ex["choice1"].strip(), ex["choice2"].strip()
            y = int(ex["label"])
            for hyp in (c1, c2):
                text = (f"{premise}\n{hyp}") if not template else template.format(premise=premise, hypothesis=hyp)
                full = _tok_ids(tok, text)
                hyp_ids = _tok_ids(tok, hyp)
                if len(full)>max_len: full = full[-max_len:]
                lab, L = _align_labels(full, hyp_ids[-max_len:])
                ids_list.append(full); labels_list.append(lab); norm_list.append(L)
            gold.append(y)
        enc = tok.pad({"input_ids": ids_list}, padding=True, max_length=max_len, return_tensors="pt")
        Lmax = enc["input_ids"].size(1)
        labels = torch.tensor([lab + [-100]*(Lmax-len(lab)) if len(lab)<Lmax else lab[:Lmax] for lab in labels_list], dtype=torch.long)
        norm = _normalize(torch.tensor(norm_list, dtype=torch.float), use_mean, alpha)
        yield enc["input_ids"], enc["attention_mask"], labels, norm, gold, 2

def iter_xstory(ds, tok, max_len, batch, use_mean, alpha):
    def extract(ex: Dict[str, Any]) -> Tuple[str,str,str,int]:
        if "context" in ex and "endings" in ex:
            ctx = " ".join(ex["context"]) if isinstance(ex["context"], list) else ex["context"]
            return ctx, ex["endings"][0], ex["endings"][1], int(ex.get("label",0))
        keys = [f"input_sentence_{i}" for i in range(1,5)]
        if all(k in ex for k in keys) and "sentence_quiz1" in ex and "sentence_quiz2" in ex:
            ctx = " ".join([ex[k] for k in keys]); lab = 0 if int(ex.get("answer_right_ending",1))==1 else 1
            return ctx, ex["sentence_quiz1"], ex["sentence_quiz2"], lab
        return ex["premise"], ex.get("choice1", ex.get("option1")), ex.get("choice2", ex.get("option2")), int(ex.get("label",0))

    for s in range(0, len(ds), batch):
        e = min(s+batch, len(ds))
        ids_list, labels_list, norm_list, gold = [], [], [], []
        for ex in ds.select(range(s, e)):
            premise, e1, e2, y = [x.strip() if isinstance(x,str) else x for x in extract(ex)]
            for endg in (e1, e2):
                text = f"{premise}\n{endg}"
                full = _tok_ids(tok, text)
                e_ids = _tok_ids(tok, endg)
                if len(full)>max_len: full = full[-max_len:]
                lab_ids, L = _align_labels(full, e_ids[-max_len:])
                ids_list.append(full); labels_list.append(lab_ids); norm_list.append(L)
            gold.append(int(y))
        enc = tok.pad({"input_ids": ids_list}, padding=True, max_length=max_len, return_tensors="pt")
        Lmax = enc["input_ids"].size(1)
        labels = torch.tensor([lab + [-100]*(Lmax-len(lab)) if len(lab)<Lmax else lab[:Lmax] for lab in labels_list], dtype=torch.long)
        norm = _normalize(torch.tensor(norm_list, dtype=torch.float), use_mean, alpha)
        yield enc["input_ids"], enc["attention_mask"], labels, norm, gold, 2

# Tatoeba retrieval helpers
@torch.no_grad()
def _encode_mean_pool(model, tok, texts: List[str], device: torch.device, fp16: bool, max_len: int, batch: int) -> torch.Tensor:
    # Preserve the sentence prefix for embeddings by temporarily switching truncation to "right".
    old = tok.truncation_side
    tok.truncation_side = "right"
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            h = model(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device),
                      output_hidden_states=True).backpack_hidden_states[-1]
        mask = enc["attention_mask"].to(device).unsqueeze(-1)  # [B, L, 1]
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        outs.append((summed / denom).float().cpu())
    tok.truncation_side = old
    return torch.cat(outs, dim=0)  # [N, D]

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    # returns a @ b^T normalized
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    sims = []
    for i in range(0, a.size(0), chunk):
        sims.append(a[i:i+chunk] @ b.T)
    return torch.cat(sims, dim=0)

def _pick(ex: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            return ex[k].strip()
    return None

def load_tatoeba_pair(src2: str, tgt2: str, split: str):
    # Use ISO-639-3 pair like "eng-tur"
    src3, tgt3 = to_iso639_3(src2), to_iso639_3(tgt2)
    lp = f"{src3}-{tgt3}"
    # Force test unless explicitly "train" or "test"
    split_fixed = "test" if split not in {"train", "test"} else split

    ds = load_dataset(
        "Helsinki-NLP/tatoeba_mt",
        split=split_fixed,
        trust_remote_code=True,
        verification_mode=VerificationMode.NO_CHECKS,
        language_pair=lp,
    )

    rows = []
    # Try the standard TSV column variants first
    src_text_candidates = [
        "source_sentence","source","src_text","source_text","sourceString","sourceText","src","text_src"
    ]
    tgt_text_candidates = [
        "target_sentence","target","tgt_text","target_text","targetString","targetText","tgt","text_tgt"
    ]
    # Some variants expose generic column names
    generic_candidates = ["sentence1", "sentence2", "text1", "text2"]

    for ex in ds:
        s = _pick(ex, src_text_candidates)
        t = _pick(ex, tgt_text_candidates)
        if s is None or t is None:
            # fallback: some loaders map to generic columns
            s = s or _pick(ex, generic_candidates[:2])
            t = t or _pick(ex, generic_candidates[2:]) or _pick(ex, generic_candidates[:2][1:])
        if s and t:
            rows.append((s, t))

    if not rows:
        # Last resort: parse tab-separated text blobs
        for ex in ds:
            txt = ex.get("text") or ex.get("data")
            if isinstance(txt, str) and "\t" in txt:
                parts = txt.split("\t")
                if len(parts) >= 4:
                    s, t = parts[-2].strip(), parts[-1].strip()
                    if s and t:
                        rows.append((s, t))

    if not rows:
        raise ValueError(f"No pairs found in Helsinki-NLP/tatoeba_mt language_pair='{lp}' split='{split_fixed}'.")

    return rows


def evaluate_tatoeba(model, tok, lang3: str, split: str, batch_size: int, max_length: int, fp16: bool,
                     device: torch.device, direction: str, file_path: Optional[str]):
    src2 = "en"
    tgt2 = map_lang("tatoeba", lang3)  # 2-letter
    if tgt2 == "en":
        raise ValueError("For tatoeba_retrieval, --lang should be a non-English target (e.g., tur/est/ind/swh).")

    # Load pairs
    pairs = load_tatoeba_pair(src2, tgt2, split=split)
    pairs = [(s.strip(), t.strip()) for s,t in pairs if isinstance(s,str) and isinstance(t,str) and s.strip() and t.strip()]
    if len(pairs) == 0:
        raise ValueError("No sentence pairs found for Tatoeba.")

    # Embeddings
    src_texts = [s for s,_ in pairs]
    tgt_texts = [t for _,t in pairs]
    src_emb = _encode_mean_pool(model, tok, src_texts, device, fp16, max_length, batch_size)
    tgt_emb = _encode_mean_pool(model, tok, tgt_texts, device, fp16, max_length, batch_size)

    # Retrieval (en → XX, optionally XX → en)
    def recall_at_1(A, B):
        sim = _cosine_sim(A, B)          # [N, N]
        nn = sim.argmax(dim=1)           # nearest B for each A
        hits = (nn == torch.arange(A.size(0))).float().mean().item()
        return hits

    r1_en2xx = recall_at_1(src_emb, tgt_emb)
    print(f"[TATOEBA en→{tgt2}/{split}] R@1: {r1_en2xx:.4f}")
    if direction == "both":
        r1_xx2en = recall_at_1(tgt_emb, src_emb)
        print(f"[TATOEBA {tgt2}→en/{split}] R@1: {r1_xx2en:.4f}")

# Evaluation entry point
def evaluate(task: str, model_path: str, lang3: str, split: str, batch_size: int, max_length: int,
             fp16: bool, device: Optional[str], subset: int, scoring: str, pmi_lambda: float,
             use_mean_nll: bool, length_alpha: float, template: Optional[str],
             use_wandb: bool, wandb_project: str, wandb_entity: Optional[str], wandb_run_name: Optional[str],
             wandb_group: Optional[str], wandb_job_type: Optional[str], 
             tatoeba_dir: str, tatoeba_file: Optional[str]):

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(model_path, dev)

    if task == "tatoeba_retrieval":
        evaluate_tatoeba(model, tok, lang3, split, batch_size, max_length, fp16, dev, tatoeba_dir, tatoeba_file)
        return

    # MCQ tasks
    lang_code = map_lang(task, lang3)
    if task == "belebele":
        ds = load_dataset("facebook/belebele", lang_code, split=split)
    elif task == "xcopa":
        ds = load_dataset("xcopa", lang_code, split=split)
    elif task == "xstorycloze":
        if split == 'test':
            split = 'eval'
            print("xstorycloze uses 'eval' split. 'test' was provided to script. Changing to 'eval'.")
        ds = load_dataset("juletxara/xstory_cloze", lang_code, split=split, trust_remote_code=True)
    else:
        raise ValueError("Unknown task: " + task)

    if subset and subset > 0:
        ds = ds.select(range(min(subset, len(ds))))

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project, entity=wandb_entity,
                name=wandb_run_name or f"{task}-{lang3}-{scoring}",
                group=wandb_group, job_type=wandb_job_type, 
                config=dict(task=task, model_path=model_path, lang=lang3, lang_code=lang_code,
                            split=split, scoring=scoring, pmi_lambda=pmi_lambda,
                            length_alpha=length_alpha, max_length=max_length)
            )
        except Exception as e:
            print(f"[warn] wandb init failed: {e}"); wb = False

    # Choose iterator
    if task == "belebele":
        iterator = iter_belebele(ds, tok, max_length, batch_size, use_mean_nll, length_alpha)
    elif task == "xcopa":
        iterator = iter_xcopa(ds, tok, max_length, batch_size, use_mean_nll, length_alpha, template)
    else:
        iterator = iter_xstory(ds, tok, max_length, batch_size, use_mean_nll, length_alpha)

    total, correct = 0, 0
    for ids, attn, labels, norm, gold, nopt in iterator:
        nll_cond = _sum_nll(model, ids, attn, labels, dev, fp16)
        s_cond = -(nll_cond / norm.to(nll_cond.device))
        if scoring == "cond":
            score = s_cond
        elif scoring == "pmi":
            # compact prior: reuse last L tokens as prior proxy
            pri_ids, pri_labels, pri_norm = [], [], []
            for k in range(ids.size(0)):
                L = int(norm[k].item()); full = ids[k].tolist()
                lab = [-100]*(len(full)-L) + full[-L:]
                pri_ids.append(full); pri_labels.append(lab); pri_norm.append(L)
            encp = tok.pad({"input_ids": pri_ids}, padding=True, max_length=max_length, return_tensors="pt")
            Lp = encp["input_ids"].size(1)
            labels_p = torch.tensor([lab + [-100]*(Lp-len(lab)) if len(lab)<Lp else lab[:Lp] for lab in pri_labels], dtype=torch.long)
            nll_p = _sum_nll(model, encp["input_ids"], encp["attention_mask"], labels_p, dev, fp16)
            s_prior = -(nll_p / torch.tensor(pri_norm, dtype=torch.float, device=nll_p.device).clamp_min(1.0))
            score = s_cond - pmi_lambda * s_prior
        else:
            raise ValueError("Unknown scoring: " + scoring)

        B = score.size(0) // nopt
        pred = score.view(B, nopt).argmax(dim=1).tolist()
        correct += sum(int(p == g) for p, g in zip(pred, gold))
        total += B

    acc = correct / max(1, total)
    print(f"[{task.upper()} {lang_code}/{split} {model_path}] {scoring} accuracy: {acc:.4f}")

    if use_wandb:
        try:
            import wandb
            wandb.log(dict(task=task, lang=lang3, split=split, scoring=scoring,
                           pmi_lambda=pmi_lambda, length_alpha=length_alpha, accuracy=acc))
            wandb.run.summary["accuracy"] = acc
            wandb.finish()
        except Exception as e:
            print(f"[warn] wandb log failed: {e}")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("Unified evaluator (Belebele / XCOPA / XStoryCloze / Tatoeba retrieval) for Backpack-GPT2.")
    ap.add_argument("--task", type=str, required=True, choices=["belebele","xcopa","xstorycloze","tatoeba_retrieval"])
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--lang", type=str, required=True, help="3-letter code (e.g., 'est', 'ind', 'swh', 'tur').")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")

    # MCQ scoring
    ap.add_argument("--scoring", type=str, default="cond", choices=["cond","pmi"])
    ap.add_argument("--pmi_lambda", type=float, default=0.1)
    ap.add_argument("--use_mean_nll", action="store_true")
    ap.add_argument("--length_alpha", type=float, default=0.0)
    ap.add_argument("--template", type=str, default=None, help="XCOPA: '{premise} ... {hypothesis}'")

    # Tatoeba retrieval options
    ap.add_argument("--tatoeba_file", type=str, default=None, help="TSV/CSV with columns 'src','tgt' (English ↔ target).")
    ap.add_argument("--tatoeba_dir", type=str, default="en2xx", choices=["en2xx","xx2en","both"])

    # wandb logging
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="eval")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_job_type", type=str, default="eval")
    ap.add_argument("--silence_warnings", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.use_mean_nll:
        args.length_alpha = 1.0
    if args.silence_warnings:
        warnings.filterwarnings('ignore')
    evaluate(
        task=args.task, model_path=args.model_path, lang3=args.lang, split=args.split,
        batch_size=args.batch_size, max_length=args.max_length, fp16=args.fp16, device=args.device,
        subset=args.subset, scoring=args.scoring, pmi_lambda=args.pmi_lambda,
        use_mean_nll=args.use_mean_nll, length_alpha=args.length_alpha,
        template=args.template,
        use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name, wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        tatoeba_dir=args.tatoeba_dir, tatoeba_file=args.tatoeba_file
    )
