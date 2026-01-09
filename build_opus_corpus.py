#!/usr/bin/env python3
# build_opus_hf_resume.py (3-letter + Hub uploader)
import argparse, os, sys, json, gzip, shutil, tempfile, hashlib, logging, time, re
from typing import List, Optional, Iterable, Iterator

from tqdm import tqdm
import langid
from sacremoses import MosesPunctNormalizer

from datasets import DatasetDict, Features, load_dataset, load_from_disk
from datasets.features import Translation

try:
    import aiohttp
except Exception:  # fall back if not available; will just skip storage_options
    aiohttp = None

# ----------------------------
# Hub target (multi-config dataset)
# ----------------------------
TARGET_REPO = "jcblaise/backpack-parallel"

# ----------------------------
# Langcode maps (639-3 <-> 639-1). Extend as needed.
# ----------------------------
# FLORES-200 three-letter (base) -> ISO-639-1 two-letter (when available)
# Value None means: no ISO-639-1 code exists for that FLORES language variety.
THREE_TO_TWO = {
    # A
    "ace": None,  # Acehnese
    "acm": "ar",  # Mesopotamian Arabic
    "acq": "ar",  # Ta’izzi-Adeni Arabic
    "aeb": "ar",  # Tunisian Arabic
    "afr": "af",
    "ajp": "ar",  # South Levantine Arabic
    "aka": "ak",
    "als": "sq",  # Tosk Albanian (macro 'sq')
    "amh": "am",
    "apc": "ar",  # North Levantine Arabic
    "arb": "ar",  # Modern Standard Arabic
    "ars": "ar",  # Najdi Arabic
    "ary": "ar",  # Moroccan Arabic
    "arz": "ar",  # Egyptian Arabic
    "asm": "as",
    "ast": None,
    "awa": None,
    "ayr": "ay",
    "azb": "az",  # South Azerbaijani (macro 'az')
    "azj": "az",  # North Azerbaijani (macro 'az')

    # B
    "bak": "ba",
    "bam": "bm",
    "ban": None,
    "bel": "be",
    "bem": None,
    "ben": "bn",
    "bho": None,
    "bjn": None,  # Banjar
    "bod": "bo",
    "bos": "bs",
    "bug": None,
    "bul": "bg",

    # C
    "cat": "ca",
    "ceb": None,
    "ces": "cs",
    "cjk": None,  # Chokwe
    "ckb": "ku",  # Central Kurdish (macro 'ku')
    "crh": None,  # Crimean Tatar
    "cym": "cy",

    # D
    "dan": "da",
    "deu": "de",
    "dik": None,  # Southwestern Dinka
    "dyu": None,  # Dyula
    "dzo": "dz",

    # E
    "ell": "el",
    "eng": "en",
    "epo": "eo",
    "est": "et",
    "eus": "eu",
    "ewe": "ee",

    # F
    "fao": "fo",
    "fij": "fj",
    "fin": "fi",
    "fon": None,
    "fra": "fr",
    "fur": None,
    "fuv": None,  # Nigerian Fulfulde (macro 'ff' exists, but variety-specific)

    # G
    "gaz": "om",  # West Central Oromo (macro 'om')
    "gla": "gd",
    "gle": "ga",
    "glg": "gl",
    "grn": "gn",
    "guj": "gu",

    # H
    "hat": "ht",
    "hau": "ha",
    "heb": "he",
    "hin": "hi",
    "hne": None,  # Chhattisgarhi
    "hrv": "hr",
    "hun": "hu",
    "hye": "hy",

    # I
    "ibo": "ig",
    "ilo": None,
    "ind": "id",
    "isl": "is",
    "ita": "it",

    # J
    "jav": "jv",
    "jpn": "ja",

    # K
    "kab": None,
    "kac": None,   # Jingpho
    "kam": None,
    "kan": "kn",
    "kas": "ks",
    "kat": "ka",
    "kbp": None,   # Kabiyè
    "kea": None,   # Kabuverdianu
    "khk": "mn",   # Halh Mongolian (macro 'mn')
    "khm": "km",
    "kik": "ki",
    "kin": "rw",
    "kir": "ky",
    "kmb": None,   # Kimbundu
    "kmr": "ku",   # Northern Kurdish (macro 'ku')
    "knc": None,   # Central Kanuri (macro 'kr' exists, variety-specific)
    "kon": "kg",   # Kongo
    "kor": "ko",
    "kaz": "kk",

    # L
    "lao": "lo",
    "lij": None,
    "lim": "li",
    "lin": "ln",
    "lit": "lt",
    "lmo": None,
    "ltg": None,   # Latgalian
    "ltz": "lb",
    "lua": None,   # Luba-Kasai (639-1 'lu' is Luba-Katanga, not Kasai)
    "lug": "lg",
    "luo": None,
    "lus": None,   # Mizo
    "lvs": "lv",   # Standard Latvian (macro 'lv')

    # M
    "mag": None,
    "mai": None,
    "mal": "ml",
    "mar": "mr",
    "min": None,   # Minangkabau
    "mkd": "mk",
    "mlt": "mt",
    "mni": None,   # Meitei
    "mos": None,   # Mossi
    "mri": "mi",
    "mya": "my",

    # N
    "nld": "nl",
    "nno": "nn",
    "nob": "nb",
    "npi": "ne",   # Nepali
    "nso": None,   # Northern Sotho (no distinct 639-1; 'st' is Southern)
    "nus": None,   # Nuer
    "nya": "ny",

    # O
    "oci": "oc",
    "ory": "or",

    # P
    "pag": None,
    "pan": "pa",
    "pap": None,   # Papiamento (no 639-1)
    "pbt": "ps",   # Southern Pashto (macro 'ps')
    "pes": "fa",   # Western Persian
    "plt": "mg",   # Plateau Malagasy (macro 'mg')
    "pol": "pl",
    "por": "pt",
    "prs": "fa",   # Dari (mapped to Persian macro)

    # Q
    "quy": "qu",   # Ayacucho Quechua (macro 'qu')

    # R
    "ron": "ro",
    "run": "rn",
    "rus": "ru",

    # S
    "sag": "sg",
    "san": "sa",
    "sat": None,   # Santali
    "scn": None,   # Sicilian
    "sco": None,   # (not in FLORES list; placeholder if needed)
    "slk": "sk",
    "slv": "sl",
    "smo": "sm",
    "sna": "sn",
    "snd": "sd",
    "som": "so",
    "sot": "st",
    "spa": "es",
    "srd": "sc",
    "srp": "sr",
    "ssw": "ss",
    "sun": "su",
    "swe": "sv",
    "swh": "sw",
    "szl": None,   # Silesian

    # T
    "tam": "ta",
    "taq": None,   # Tamasheq (Tfng/Latn variants in FLORES; no 639-1)
    "tat": "tt",
    "tel": "te",
    "tgk": "tg",
    "tgl": "tl",
    "tha": "th",
    "tir": "ti",
    "tpi": None,   # Tok Pisin
    "tsn": "tn",
    "tso": "ts",
    "tuk": "tk",
    "tum": None,
    "tur": "tr",
    "twi": None,   # Twi (historic 'tw' deprecated; prefer None)

    # U
    "uig": "ug",
    "ukr": "uk",
    "umb": None,
    "urd": "ur",
    "uzn": "uz",

    # V
    "vec": None,
    "vie": "vi",

    # W
    "war": None,
    "wol": "wo",

    # X
    "xho": "xh",

    # Y
    "ydd": "yi",   # Eastern Yiddish -> Yiddish
    "yor": "yo",
    "yue": None,   # Yue (Cantonese) no distinct 639-1 (don’t coerce to 'zh')

    # Z
    "zho": "zh",
    "zsm": "ms",   # Standard Malay
    "zul": "zu",
}

TWO_TO_THREE = {v: k for k, v in THREE_TO_TWO.items()}

# ----------------------------
# OpusTools (pure Python)
# ----------------------------
try:
    from opustools.opus_read import OpusRead
except Exception:
    OpusRead = None

# Optional: LaBSE
try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    torch = None

# Trimmed, high bang-for-buck default corpora; customize as you like
OPUS_CORPORA_DEFAULT = [
    "JW300", "WikiMatrix", "TED2020", "GlobalVoices", "QED", "CCAligned",
]

LOGGER = logging.getLogger("opus_resume")

# ----------------------------
# Logging
# ----------------------------
def setup_logging(level="INFO", log_file: Optional[str] = None):
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(LOGGER.level); sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(LOGGER.level); fh.setFormatter(fmt); LOGGER.addHandler(fh)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.INFO)

# ----------------------------
# Atomic I/O helpers (jsonl.gz with sidecars)
# ----------------------------
def _sidecar(path: str) -> str:
    return path + ".done.json"

def _is_complete(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0 and os.path.exists(_sidecar(path))

def _write_sidecar(path: str, lines: int):
    side = _sidecar(path)
    data = {"lines": int(lines), "completed_at": time.time()}
    tmp = side + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, side)

def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    try:
        side = _sidecar(path)
        if os.path.exists(side):
            os.remove(side)
    except Exception:
        pass

def atomic_write_jsonl_gz(path: str, rows: Iterable[dict], desc: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for obj in tqdm(rows, desc=desc, unit="row"):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    with open(tmp, "rb") as fcheck:
        os.fsync(fcheck.fileno())
    os.replace(tmp, path)
    _write_sidecar(path, count)
    return count

def append_jsonl_gz(path: str, rows: Iterable[dict], desc: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        for obj in tqdm(rows, desc=desc, unit="row"):
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def file_exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

# ----------------------------
# OPUS extraction + raw cache
# ----------------------------
def opus_fetch_pairs_cached(
    corpus: str, src2: str, tgt2: str, tmpdir: str,
    raw_path: str, max_samples: int = 0,
    root_directory: Optional[str] = None,
    download_dir: Optional[str] = None,
    preprocess: str = "moses",
):
    if _is_complete(raw_path):
        LOGGER.info("[%s] raw cache present ✓ → %s", corpus, raw_path)
        return
    if file_exists_nonempty(raw_path) and not _is_complete(raw_path):
        LOGGER.warning("[%s] raw cache incomplete; rebuilding …", corpus)
        _safe_remove(raw_path)

    if OpusRead is None:
        LOGGER.error("opustools not installed. `pip install opustools`.")
        sys.exit(1)

    out_src = os.path.join(tmpdir, f"{corpus}.{src2}")
    out_tgt = os.path.join(tmpdir, f"{corpus}.{tgt2}")

    kwargs = dict(
        directory=corpus, source=src2, target=tgt2,
        preprocess=preprocess, maximum=(max_samples or None),
        write=[out_src, out_tgt], write_mode="moses", suppress_prompts=True,
    )
    if root_directory: kwargs["root_directory"] = root_directory
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)
        kwargs["download_dir"] = download_dir

    LOGGER.info("[OPUS] %s: fetching via OpusRead …", corpus)
    try:
        OpusRead(**kwargs).printPairs()
    except Exception as e:
        LOGGER.warning("[OPUS] failed for %s: %s", corpus, str(e)[:400])
        return

    if not (os.path.exists(out_src) and os.path.exists(out_tgt)):
        LOGGER.warning("[OPUS] no output for %s", corpus); return

    def gen():
        with open(out_src, "r", encoding="utf-8", errors="ignore") as fs, \
             open(out_tgt, "r", encoding="utf-8", errors="ignore") as ft:
            for s, t in zip(fs, ft):
                s = s.strip(); t = t.strip()
                if s and t:
                    yield {"src": s, "tgt": t}

    try:
        atomic_write_jsonl_gz(raw_path, gen(), desc=f"{corpus}: write raw")
    finally:
        for p in (out_src, out_tgt):
            try: os.remove(p)
            except Exception: pass

# ----------------------------
# HF Hub extraction + raw cache
# ----------------------------

def _lang_key_variants_3_and_2(lang3: str) -> List[str]:
    """
    Return likely keys seen under 'translation' for this language across HF datasets.
    Covers 639-3 (eng), FLORES script tags (eng_Latn), and 639-1 (en) when available.
    """
    keys = {lang3, f"{lang3}_Latn"}
    lang2 = THREE_TO_TWO.get(lang3)
    if lang2:
        keys.add(lang2)
        keys.add(f"{lang2}_Latn")
    return list(keys)

def _candidate_hf_configs(src3: str, tgt3: str) -> List[str]:
    """
    Try common config name patterns, ordered from most typical for FLORES-centric sets.
    e.g., eng_Latn-swh_Latn, eng-swh, en-sw
    """
    out = []
    s3, t3 = src3, tgt3
    s2, t2 = THREE_TO_TWO.get(src3), THREE_TO_TWO.get(tgt3)

    # FLORES-200 style (Latin script)
    out.append(f"{s3}_Latn-{t3}_Latn")
    # Plain 3-letter
    out.append(f"{s3}-{t3}")
    # Mixed fallbacks (rare, but cheap to try)
    out.append(f"{s3}_Latn-{t3}")
    out.append(f"{s3}-{t3}_Latn")

    # 2-letter configs, if they exist on a builder
    if s2 and t2:
        out.append(f"{s2}-{t2}")
        out.append(f"{s2}_Latn-{t2}_Latn")

    # Deduplicate while preserving order
    seen = set(); ordered = []
    for c in out:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def _hf_try_load_dataset(ds_id: str, src3: str, tgt3: str, split: str, timeout_sec: int):
    """
    Try to load a HF dataset with a handful of plausible config names.
    Returns (dataset, used_config) or (None, None) if all attempts fail.
    """
    storage_options = None
    if aiohttp is not None:
        storage_options = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=timeout_sec)}}

    # Try with candidate configs first (many builders require a pair config)
    for cfg in _candidate_hf_configs(src3, tgt3):
        try:
            d = load_dataset(ds_id, cfg, split=split, storage_options=storage_options) if storage_options \
                else load_dataset(ds_id, cfg, split=split)
            LOGGER.info("[HF] loaded %s (config=%s, split=%s)", ds_id, cfg, split)
            return d, cfg
        except Exception as e:
            LOGGER.debug("[HF] %s config=%s failed: %s", ds_id, cfg, str(e)[:200])

    # Last resort: some builders have a single global config
    try:
        d = load_dataset(ds_id, split=split, storage_options=storage_options) if storage_options \
            else load_dataset(ds_id, split=split)
        LOGGER.info("[HF] loaded %s (no explicit config, split=%s)", ds_id, split)
        return d, "default"
    except Exception as e:
        LOGGER.warning("[HF] failed to load %s (all configs tried). Last error: %s", ds_id, str(e)[:300])
        return None, None

def hf_fetch_pairs_cached(
    hf_id: str,
    src3: str, tgt3: str,
    corp_dir: str,
    split: str = "train",
    timeout_sec: int = 3600,
) -> Optional[str]:
    """
    Loads a HF dataset and writes it to a raw cache jsonl.gz with {'src','tgt'} to align
    with the OPUS flow. Returns a *sanitized alias* (file prefix) to be used downstream
    for filtering/merge, or None if it failed.
    """
    d, used_cfg = _hf_try_load_dataset(hf_id, src3, tgt3, split, timeout_sec)
    if d is None:
        return None

    # We'll build a safe file stem like: hf__allenai_nllb__eng_Latn-swh_Latn
    safe_ds = re.sub(r"[^A-Za-z0-9_.-]+", "_", hf_id)
    safe_cfg = re.sub(r"[^A-Za-z0-9_.-]+", "_", used_cfg or "default")
    alias = f"hf__{safe_ds}__{safe_cfg}"

    raw_path = os.path.join(corp_dir, f"{alias}.raw.jsonl.gz")
    if _is_complete(raw_path):
        LOGGER.info("[%s] raw cache present ✓ → %s", alias, raw_path)
        return alias
    if file_exists_nonempty(raw_path) and not _is_complete(raw_path):
        LOGGER.warning("[%s] raw cache incomplete; rebuilding …", alias)
        _safe_remove(raw_path)

    src_keys = _lang_key_variants_3_and_2(src3)
    tgt_keys = _lang_key_variants_3_and_2(tgt3)

    def gen_rows():
        # Expect common "translation" schema with language-keyed dict
        # e.g. {'translation': {'eng_Latn': '...', 'swh_Latn': '...'}}
        for ex in d:
            tr = ex.get("translation")
            if not isinstance(tr, dict):
                continue
            s = next((tr[k] for k in src_keys if k in tr and isinstance(tr.get(k), str) and tr[k]), None)
            t = next((tr[k] for k in tgt_keys if k in tr and isinstance(tr.get(k), str) and tr[k]), None)
            if s and t:
                yield {"src": s, "tgt": t}

    try:
        atomic_write_jsonl_gz(raw_path, gen_rows(), desc=f"{alias}: write raw")
        return alias
    except Exception as e:
        LOGGER.warning("[HF] failed writing raw for %s: %s", alias, str(e)[:300])
        _safe_remove(raw_path)
        return None

# ----------------------------
# Filters + filtered cache
# ----------------------------
def simple_filters_iter(
    rows: Iterable[dict],
    min_len=1, max_len=200,
    len_ratio_low=0.5, len_ratio_high=2.0,
    lang_src2="en", lang_tgt2="id",
    normalize_punct=True,
) -> Iterator[dict]:
    mpn = MosesPunctNormalizer() if normalize_punct else None
    for r in rows:
        s, t = r["src"], r["tgt"]
        if mpn:
            s = mpn.normalize(s); t = mpn.normalize(t)
        sw, tw = s.split(), t.split()
        if not (min_len <= len(sw) <= max_len): continue
        if not (min_len <= len(tw) <= max_len): continue
        ratio = len(sw) / max(1, len(tw))
        if not (len_ratio_low <= ratio <= len_ratio_high):
            inv = 1.0 / ratio
            if not (len_ratio_low <= inv <= len_ratio_high): continue
        ls, _ = langid.classify(s); lt, _ = langid.classify(t)
        if ls != lang_src2 or lt != lang_tgt2: continue
        yield {"src": s, "tgt": t}

def ensure_filtered_cached(
    corpus: str, raw_path: str, filtered_path: str,
    **filter_kwargs
):
    if _is_complete(filtered_path):
        LOGGER.info("[%s] filtered cache present ✓ → %s", corpus, filtered_path)
        return
    if not _is_complete(raw_path):
        if file_exists_nonempty(raw_path):
            LOGGER.warning("[%s] raw cache incomplete; removing …", corpus)
            _safe_remove(raw_path)
        LOGGER.info("[%s] no valid raw cache; skipping filter for now.", corpus)
        return

    rows = read_jsonl_gz(raw_path)
    try:
        atomic_write_jsonl_gz(
            filtered_path,
            simple_filters_iter(rows, **filter_kwargs),
            desc=f"{corpus}: filter"
        )
    except Exception as e:
        LOGGER.warning("[%s] filter write failed, cleaning partial: %s", corpus, e)
        _safe_remove(filtered_path)
        raise

# ----------------------------
# Merge + dedup
# ----------------------------
def sha_pair(s: str, t: str) -> str:
    return hashlib.sha1(f"{s}\t{t}".encode("utf-8")).hexdigest()

def build_merged_dedup(corpora: List[str], filtered_dir: str, merged_path: str):
    if _is_complete(merged_path):
        LOGGER.info("Merged+dedup present ✓ → %s", merged_path)
        return

    def gen():
        seen = set()
        for c in corpora:
            fp = os.path.join(filtered_dir, f"{c}.filtered.jsonl.gz")
            if not _is_complete(fp):
                continue
            try:
                for r in read_jsonl_gz(fp):
                    h = sha_pair(r["src"], r["tgt"])
                    if h in seen: continue
                    seen.add(h)
                    yield r
            except Exception as e:
                LOGGER.warning("[%s] filtered cache seems corrupt; removing: %s", c, e)
                _safe_remove(fp)

    atomic_write_jsonl_gz(merged_path, gen(), desc="Merge + dedup")

# ----------------------------
# LaBSE (resumable)
# ----------------------------
def labse_resumable(
    merged_path: str, kept_path: str, progress_path: str,
    threshold: float = 0.8, batch_size: int = 512, device: Optional[str] = None
):
    # Ensure sane kept file if old artifact exists without sidecar
    if file_exists_nonempty(kept_path) and not os.path.exists(_sidecar(kept_path)):
        LOGGER.warning("Kept file exists without sidecar; truncating to restart appends: %s", kept_path)
        _safe_remove(kept_path)

    if SentenceTransformer is None:
        LOGGER.warning("sentence-transformers not installed; skipping LaBSE (pass-through).")
        if not _is_complete(kept_path):
            count = 0
            with gzip.open(merged_path, "rt", encoding="utf-8") as src, \
                 gzip.open(kept_path, "wt", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line); count += 1
            _write_sidecar(kept_path, count)
        return

    device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    # Progress
    processed = 0
    if os.path.exists(progress_path):
        try:
            processed = json.load(open(progress_path))["processed"]
        except Exception:
            processed = 0

    # Count total
    total = 0
    with gzip.open(merged_path, "rt", encoding="utf-8") as f:
        for _ in f: total += 1
    LOGGER.info("LaBSE resume: processed=%d / total=%d", processed, total)

    def stream_from_offset(path: str, offset: int) -> Iterator[dict]:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset: continue
                s = line.strip()
                if s:
                    yield json.loads(s)

    cursor = processed
    batch_src, batch_tgt, buffer_rows = [], [], []

    def flush():
        nonlocal batch_src, batch_tgt, buffer_rows, cursor
        if not batch_src: return
        emb_s = model.encode(batch_src, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        emb_t = model.encode(batch_tgt, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
        sims = (emb_s * emb_t).sum(dim=1)
        mask_l = sims >= threshold
        kept_rows = []
        mask_l = mask_l.detach().to("cpu").tolist()
        for row, keep in zip(buffer_rows, mask_l):
            if keep:
                kept_rows.append({"src": row["src"], "tgt": row["tgt"]})
        append_jsonl_gz(kept_path, kept_rows, desc="LaBSE append (batch)")
        cursor += len(buffer_rows)
        json.dump({"processed": cursor}, open(progress_path, "w"))
        batch_src.clear(); batch_tgt.clear(); buffer_rows.clear()

    for row in tqdm(stream_from_offset(merged_path, processed), total=(total - processed), desc="LaBSE resume", unit="row"):
        s, t = row["src"], row["tgt"]
        batch_src.append(s); batch_tgt.append(t); buffer_rows.append(row)
        if len(batch_src) >= batch_size:
            flush()
    flush()

    # Write sidecar for kept file (completion marker)
    kept_lines = 0
    if file_exists_nonempty(kept_path):
        try:
            for _ in read_jsonl_gz(kept_path):
                kept_lines += 1
        except Exception:
            LOGGER.warning("Kept file corrupt; removing for next run.")
            _safe_remove(kept_path)
            kept_lines = 0
    if kept_lines > 0:
        _write_sidecar(kept_path, kept_lines)

# ----------------------------
# Final HF write (LOW-RAM) → returns DatasetDict
# ----------------------------
def write_hf_dataset_dir(kept_path: str, out_dir: str, src3: str, tgt3: str,
                         target_size: int, shuffle_seed: int) -> DatasetDict:
    """
    Build a HF dataset on disk with 'translation' using 3-letter keys.
    Returns the DatasetDict (memory-mapped; safe to push_to_hub).
    """
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset("json", data_files=kept_path, split="train")

    ds = ds.map(
        lambda ex: {"translation": [{src3: s, tgt3: t} for s, t in zip(ex["src"], ex["tgt"])]},
        batched=True,
        remove_columns=["src", "tgt"]
    )

    ds = ds.shuffle(seed=shuffle_seed)
    if target_size and len(ds) > target_size:
        ds = ds.select(range(target_size))

    ds = ds.cast(Features({"translation": Translation(languages=[src3, tgt3])}))
    dset = DatasetDict({"train": ds})
    LOGGER.info("[HF] Saving to %s", out_dir)
    dset.save_to_disk(out_dir)
    LOGGER.info("[HF] Done → load_from_disk('%s')", out_dir)
    return dset

# ----------------------------
# Pipeline (restartable)
# ----------------------------
def corpus_pipeline(
    corpora: List[str], src3: str, tgt3: str, hf_out_dir: str,
    work_dir: str,
    max_per_corpus: int = 0,
    min_len: int = 4, max_len: int = 64,
    len_ratio_low: float = 0.67, len_ratio_high: float = 1.5,
    labse_threshold: float = 0.82,
    batch_size: int = 512,
    target_size: int = 2_000_000,
    shuffle_seed: int = 13,
    opus_root: Optional[str] = None,
    opus_download_dir: Optional[str] = None,
    preprocess: str = "moses",
    cleanup_work_dir_on_success: bool = False,
    hf_corpora: Optional[List[str]] = None,
    hf_corpora_split: str = "train",
    hf_timeout_sec: int = 3600,
) -> DatasetDict:
    # map to 2-letter for OPUS/langid
    if src3 not in THREE_TO_TWO or tgt3 not in THREE_TO_TWO:
        raise ValueError(f"Unsupported 3-letter langcode(s): {src3}, {tgt3}")
    src2, tgt2 = THREE_TO_TWO[src3], THREE_TO_TWO[tgt3]

    os.makedirs(work_dir, exist_ok=True)
    corp_dir = os.path.join(work_dir, "corpora")
    filt_dir = corp_dir
    merged_dir = os.path.join(work_dir, "merged")
    labse_dir = os.path.join(work_dir, "labse")
    os.makedirs(corp_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(labse_dir, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix=f"opus_{src2}{tgt2}_")
    LOGGER.info("Using temp dir: %s", tmpdir)
    try:
        # 1) Fetch raw
        for c in tqdm(corpora, desc="Fetch corpora", unit="corpus"):
            raw_path = os.path.join(corp_dir, f"{c}.raw.jsonl.gz")
            opus_fetch_pairs_cached(
                c, src2, tgt2, tmpdir, raw_path, max_samples=max_per_corpus,
                root_directory=opus_root, download_dir=opus_download_dir, preprocess=preprocess
            )

        # 1a) (Optional) Fetch HF corpora into the same raw cache
        hf_aliases = []
        if hf_corpora:
            for hf_id in tqdm(hf_corpora, desc="Fetch HF corpora", unit="dataset"):
                alias = hf_fetch_pairs_cached(
                    hf_id,
                    src3=src3, tgt3=tgt3,
                    corp_dir=corp_dir,
                    split=hf_corpora_split,
                    timeout_sec=hf_timeout_sec,
                )
                if alias:
                    hf_aliases.append(alias)

        all_corpora = list(corpora) + hf_aliases

        # 2) Filter per-corpus
        filter_kwargs = dict(
            min_len=min_len, max_len=max_len,
            len_ratio_low=len_ratio_low, len_ratio_high=len_ratio_high,
            lang_src2=src2, lang_tgt2=tgt2, normalize_punct=True
        )
        for c in tqdm(all_corpora, desc="Filter corpora", unit="corpus"):
            raw_path = os.path.join(corp_dir, f"{c}.raw.jsonl.gz")
            filtered_path = os.path.join(filt_dir, f"{c}.filtered.jsonl.gz")
            ensure_filtered_cached(c, raw_path, filtered_path, **filter_kwargs)

        # 3) Merge + dedup
        merged_path = os.path.join(merged_dir, "merged_dedup.jsonl.gz")
        build_merged_dedup(all_corpora, filt_dir, merged_path)

        # 4) LaBSE resumable
        kept_path = os.path.join(labse_dir, "kept.jsonl.gz")
        progress_path = os.path.join(labse_dir, "progress.json")
        if labse_threshold > 0.0:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            labse_resumable(
                merged_path, kept_path, progress_path,
                threshold=labse_threshold, batch_size=batch_size, device=device
            )
        else:
            if not _is_complete(kept_path):
                LOGGER.info("LaBSE disabled; copying merged to kept …")
                cnt = 0
                with gzip.open(merged_path, "rt", encoding="utf-8") as srcf, \
                     gzip.open(kept_path, "wt", encoding="utf-8") as dstf:
                    for line in srcf: dstf.write(line); cnt += 1
                _write_sidecar(kept_path, cnt)

        # 5) HF write (low-RAM) with 3-letter keys
        dset = write_hf_dataset_dir(kept_path, hf_out_dir, src3, tgt3, target_size, shuffle_seed)

        if cleanup_work_dir_on_success:
            LOGGER.info("Cleanup requested: removing work_dir %s", work_dir)
            shutil.rmtree(work_dir, ignore_errors=True)

        return dset
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        LOGGER.info("Cleaned temp dir: %s", tmpdir)

# ----------------------------
# Hub helpers
# ----------------------------
def push_to_hub(dset: DatasetDict, config_name: str, token: Optional[str] = None,
                max_shard_size: str = "1GB"):
    LOGGER.info("Pushing to Hub: repo=%s, config=%s", TARGET_REPO, config_name)
    dset.push_to_hub(
        TARGET_REPO,
        config_name=config_name,
        token=token,
        max_shard_size=max_shard_size,
    )
    LOGGER.info("✓ Pushed: %s [%s]", TARGET_REPO, config_name)

def upload_dir_to_hub(saved_dir: str, config_name: str, token: Optional[str] = None,
                      max_shard_size: str = "1GB"):
    LOGGER.info("Loading saved dataset from %s …", saved_dir)
    dset = load_from_disk(saved_dir)
    push_to_hub(dset, config_name=config_name, token=token, max_shard_size=max_shard_size)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        "Assemble OPUS bitext with checkpointing/resume, output HF dataset (3-letter keys)."
    )

    # Languages: now 3-letter on the CLI
    ap.add_argument("--src", type=str, default="eng", help="Source language (ISO 639-3, e.g., eng)")
    ap.add_argument("--tgt", type=str, default="ind", help="Target language (ISO 639-3, e.g., ind)")

    ap.add_argument("--hf_out", type=str, required=True, help="Output directory for Hugging Face dataset.")
    ap.add_argument("--work_dir", type=str, required=True, help="Persistent checkpoint dir (e.g., $SCRATCH/opus_work/eng-ind).")
    ap.add_argument("--corpora", type=str, nargs="*", default=OPUS_CORPORA_DEFAULT)
    ap.add_argument("--max_per_corpus", type=int, default=0)

    # HF Hub extras
    ap.add_argument("--hf_corpora", type=str, nargs="*", default=[],
                    help="List of HF dataset IDs to include (e.g., allenai/nllb allenai/wmt22_african).")
    ap.add_argument("--hf_corpora_split", type=str, default="train",
                    help="Which split to load from HF corpora (default: train).")
    ap.add_argument("--hf_timeout_sec", type=int, default=3600,
                    help="aiohttp total timeout seconds for HF downloads (large builders like nllb).")

    # Filters
    ap.add_argument("--min_len", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--len_ratio_low", type=float, default=0.67)
    ap.add_argument("--len_ratio_high", type=float, default=1.5)

    # LaBSE
    ap.add_argument("--labse_threshold", type=float, default=0.82)
    ap.add_argument("--batch_size", type=int, default=512)

    # Final shaping
    ap.add_argument("--target_size", type=int, default=2_000_000)
    ap.add_argument("--shuffle_seed", type=int, default=13)

    # OpusTools config
    ap.add_argument("--opus_root", type=str, default="", help="Optional local OPUS mirror root.")
    ap.add_argument("--opus_download_dir", type=str, default="opus_cache", help="Cache dir for OPUS downloads.")
    ap.add_argument("--preprocess", type=str, default="moses", choices=["moses","raw","xml"])

    # Cleanup flag
    ap.add_argument("--cleanup_work_dir_on_success", action="store_true",
                    help="If set, deletes --work_dir after a successful run.")

    # Logging
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--log_file", type=str, default="")

    # NEW: Hub upload controls
    ap.add_argument("--upload_data", action="store_true",
                    help="If set, push the built dataset to the Hub as a new config of jcblaise/backpack-parallel.")
    ap.add_argument("--config_name", type=str, default="",
                    help="Config name for the Hub (e.g., 'eng-spa'). Defaults to '{src}-{tgt}'.")
    ap.add_argument("--push_max_shard_size", type=str, default="1GB",
                    help="Shard size for push_to_hub (e.g., 500MB, 1GB).")
    ap.add_argument("--upload_from_disk", type=str, default="",
                    help="(Optional) Path to a previously saved dataset dir to upload to Hub as --config_name (skips building).")

    return ap.parse_args()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file or None)

    # Fast path: upload a previously saved directory
    if args.upload_from_disk:
        cfg = args.config_name.strip() or f"{args.src}-{args.tgt}"
        token = os.environ.get("HF_TOKEN", None)
        upload_dir_to_hub(args.upload_from_disk, cfg, token=token, max_shard_size=args.push_max_shard_size)
        sys.exit(0)

    # Validate and normalize lang codes (3-letter on CLI)
    src3 = args.src.strip().lower()
    tgt3 = args.tgt.strip().lower()
    if src3 not in THREE_TO_TWO or tgt3 not in THREE_TO_TWO:
        LOGGER.error("Unsupported 3-letter langcode(s): %s, %s", src3, tgt3)
        sys.exit(2)

    LOGGER.info("Starting (resume-enabled): %s→%s | corpora=%d", src3, tgt3, len(args.corpora))
    try:
        dset = corpus_pipeline(
            corpora=args.corpora,
            src3=src3, tgt3=tgt3,
            hf_out_dir=args.hf_out,
            work_dir=args.work_dir,
            max_per_corpus=args.max_per_corpus,
            min_len=args.min_len, max_len=args.max_len,
            len_ratio_low=args.len_ratio_low, len_ratio_high=args.len_ratio_high,
            labse_threshold=args.labse_threshold,
            batch_size=args.batch_size,
            target_size=args.target_size,
            shuffle_seed=args.shuffle_seed,
            opus_root=(args.opus_root or None),
            opus_download_dir=(args.opus_download_dir or None),
            preprocess=args.preprocess,
            cleanup_work_dir_on_success=args.cleanup_work_dir_on_success,
            hf_corpora=(args.hf_corpora or []),
            hf_corpora_split=args.hf_corpora_split,
            hf_timeout_sec=args.hf_timeout_sec,
        )
        LOGGER.info("Finished building dataset on disk: %s", args.hf_out)

        if args.upload_data:
            cfg = args.config_name.strip() or f"{src3}-{tgt3}"
            token = os.environ.get("HF_TOKEN", None)
            push_to_hub(dset, config_name=cfg, token=token, max_shard_size=args.push_max_shard_size)
            LOGGER.info("You can now load it via: load_dataset('%s', '%s')", TARGET_REPO, cfg)
        else:
            LOGGER.info("Upload skipped (use --upload_data to push). To upload later:")
            LOGGER.info("  python build_opus_hf_resume.py --upload_from_disk %s --config_name %s", args.hf_out, (args.config_name or f"{src3}-{tgt3}"))

    except Exception:
        LOGGER.exception("Fatal error.")
        sys.exit(1)
