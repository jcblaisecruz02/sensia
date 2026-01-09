#!/usr/bin/env python
"""Compute FLORES-200 perplexity for Backpack models."""

import argparse, math
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


def prepare_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, fp16=False, max_length=256, batch_size=32):
    nll_sum, tok_count = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
        batch = texts[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
            loss = out.loss

        valid_tokens = (labels != -100).sum().item()
        nll_sum += loss.item() * max(1, valid_tokens)
        tok_count += valid_tokens

    ppl = math.exp(nll_sum / max(1, tok_count))
    return ppl


def evaluate_flores(model_path, lang="sw", split="devtest",
                    batch_size=32, max_length=256, device="cuda", fp16=False):
    device = torch.device(device)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = prepare_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("facebook/flores", lang, split=split)
    texts = ds["sentence"]
    ppl = compute_perplexity(model, tokenizer, texts, device, fp16, max_length, batch_size)
    print(f"[FLORES {lang}/{split}] Perplexity: {ppl:.2f}")
    return ppl


def parse_args():
    p = argparse.ArgumentParser(description="Compute perplexity on FLORES sentences.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--lang", type=str, default="sw", help="FLORES language code, e.g. eng_Latn, swh_Latn, etc.")
    p.add_argument("--split", type=str, default="devtest", choices=["dev", "devtest"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_flores(args.model_path, lang=args.lang,
                    split=args.split, batch_size=args.batch_size,
                    max_length=args.max_length, device=args.device,
                    fp16=args.fp16)