from datasets import load_dataset, Dataset as HFDataset

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from .helpers import get_world_size

FLORES_CODE_MAP = {
    # A–E
    "afr": "afr_Latn",
    "amh": "amh_Ethi",
    "ara": "arb_Arab",
    "asm": "asm_Beng",
    "ast": "ast_Latn",
    "bel": "bel_Cyrl",
    "ben": "ben_Beng",
    "bod": "bod_Tibt",
    "bos": "bos_Latn",
    "bul": "bul_Cyrl",
    "cat": "cat_Latn",
    "ceb": "ceb_Latn",
    "ces": "ces_Latn",
    "ckb": "ckb_Arab",
    "cym": "cym_Latn",
    "dan": "dan_Latn",
    "deu": "deu_Latn",
    "ell": "ell_Grek",
    "eng": "eng_Latn",
    "epo": "epo_Latn",
    "est": "est_Latn",
    "eus": "eus_Latn",

    # F–K
    "fao": "fao_Latn",
    "fas": "pes_Arab",
    "fin": "fin_Latn",
    "fra": "fra_Latn",
    "fry": "fry_Latn",
    "gle": "gle_Latn",
    "glg": "glg_Latn",
    "guj": "guj_Gujr",
    "hau": "hau_Latn",
    "heb": "heb_Hebr",
    "hin": "hin_Deva",
    "hrv": "hrv_Latn",
    "hun": "hun_Latn",
    "hye": "hye_Armn",
    "ibo": "ibo_Latn",
    "ind": "ind_Latn",
    "isl": "isl_Latn",
    "ita": "ita_Latn",
    "jav": "jav_Latn",
    "jpn": "jpn_Jpan",
    "kan": "kan_Knda",
    "kat": "kat_Geor",
    "kaz": "kaz_Cyrl",
    "khm": "khm_Khmr",
    "kir": "kir_Cyrl",
    "kor": "kor_Hang",

    # L–S
    "lao": "lao_Laoo",
    "lat": "lat_Latn",
    "lav": "lav_Latn",
    "lin": "lin_Latn",
    "lit": "lit_Latn",
    "lug": "lug_Latn",
    "mal": "mal_Mlym",
    "mar": "mar_Deva",
    "mkd": "mkd_Cyrl",
    "mlt": "mlt_Latn",
    "mri": "mri_Latn",
    "msa": "zsm_Latn",
    "mya": "mya_Mymr",
    "nep": "npi_Deva",
    "nld": "nld_Latn",
    "nno": "nno_Latn",
    "nob": "nob_Latn",
    "nya": "nya_Latn",
    "oci": "oci_Latn",
    "ori": "ory_Orya",
    "pan": "pan_Guru",
    "pol": "pol_Latn",
    "por": "por_Latn",
    "pus": "pbt_Arab",
    "ron": "ron_Latn",
    "run": "run_Latn",
    "rus": "rus_Cyrl",
    "slk": "slk_Latn",
    "slv": "slv_Latn",
    "sna": "sna_Latn",
    "som": "som_Latn",
    "spa": "spa_Latn",
    "srp": "srp_Cyrl",
    "swe": "swe_Latn",
    "swh": "swh_Latn",
    "tam": "tam_Taml",
    "tel": "tel_Telu",
    "tha": "tha_Thai",
    "tur": "tur_Latn",
    "ukr": "ukr_Cyrl",
    "urd": "urd_Arab",
    "uzb": "uzn_Latn",
    "vie": "vie_Latn",
    "wol": "wol_Latn",
    "yor": "yor_Latn",
    "zho": "zho_Hans",
    "zho_trad": "zho_Hant",
}

def preprocess_function(example, tokenizer, src, tgt, max_length):
    """Tokenize a translation pair and return model-ready dict with src/tgt ids and masks."""
    src_text = example["translation"][src]
    tgt_text = example["translation"][tgt]

    src_enc = tokenizer(
        src_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    tgt_enc = tokenizer(
        tgt_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    return {
        "input_ids_src": src_enc["input_ids"],
        "attention_mask_src": src_enc["attention_mask"],
        "input_ids_tgt": tgt_enc["input_ids"],
        "attention_mask_tgt": tgt_enc["attention_mask"],
    }

class ContrastiveDataset(Dataset):
    """Simple wrapper around tokenized parallel data for contrastive training."""

    def __init__(self, data):
        self.data = data  # data should be a dict of lists

    def __len__(self):
        """Return dataset size (matches number of source sequences)."""
        return len(self.data["input_ids_src"])

    def __getitem__(self, idx):
        """Fetch a single example as torch tensors for src/tgt ids and masks."""
        return {
            "input_ids_src": torch.tensor(self.data["input_ids_src"][idx]),
            "attention_mask_src": torch.tensor(self.data["attention_mask_src"][idx]),
            "input_ids_tgt": torch.tensor(self.data["input_ids_tgt"][idx]),
            "attention_mask_tgt": torch.tensor(self.data["attention_mask_tgt"][idx]),
        }

def collate_fn(batch, tokenizer):
    """Pad variable-length src/tgt sequences in a batch using the tokenizer pad method."""
    src_input_ids = [item["input_ids_src"] for item in batch]
    src_attention = [item["attention_mask_src"] for item in batch]
    tgt_input_ids = [item["input_ids_tgt"] for item in batch]
    tgt_attention = [item["attention_mask_tgt"] for item in batch]

    src_batch = tokenizer.pad({"input_ids": src_input_ids, "attention_mask": src_attention}, return_tensors="pt")
    tgt_batch = tokenizer.pad({"input_ids": tgt_input_ids, "attention_mask": tgt_attention}, return_tensors="pt")

    return {
        "input_ids_src": src_batch["input_ids"],
        "attention_mask_src": src_batch["attention_mask"],
        "input_ids_tgt": tgt_batch["input_ids"],
        "attention_mask_tgt": tgt_batch["attention_mask"],
    }

def load_flores_pair(src_code: str, tgt_code: str, split: str = "devtest", dataset: str = "facebook/flores"):
    """Load two FLORES languages, align them by id, and return a parallel HF Dataset."""
    if 'flores_plus' in dataset:
        if tgt_code == 'est_Latn': tgt_code = 'ekk_Latn'

    ds_src = load_dataset(dataset, src_code, split=split)
    ds_tgt = load_dataset(dataset, tgt_code, split=split)

    # Align by 'id' (FLORES gives consistent ids across languages)
    ds_src = ds_src.sort("id")
    ds_tgt = ds_tgt.sort("id")
    assert len(ds_src) == len(ds_tgt), "Mismatched sizes between FLORES languages"

    # Strong integrity check
    if ds_src["id"] != ds_tgt["id"]:
        raise ValueError("FLORES ids are not aligned; check versions or sorting.")
    
    if 'flores_plus' in dataset:
        return HFDataset.from_dict({
            "id": ds_src["id"],
            "src_sentence": ds_src["text"],
            "tgt_sentence": ds_tgt["text"],
        }) 

    return HFDataset.from_dict({
        "id": ds_src["id"],
        "src_sentence": ds_src["sentence"],
        "tgt_sentence": ds_tgt["sentence"],
    })

def preprocess_flores(example, tokenizer, max_length):
    """Tokenize FLORES sentence pairs into src/tgt ids and attention masks."""
    src_text = example["src_sentence"]
    tgt_text = example["tgt_sentence"]

    src_enc = tokenizer(src_text, truncation=True, max_length=max_length)
    tgt_enc = tokenizer(tgt_text, truncation=True, max_length=max_length)

    return {
        "input_ids_src": src_enc["input_ids"],
        "attention_mask_src": src_enc["attention_mask"],
        "input_ids_tgt": tgt_enc["input_ids"],
        "attention_mask_tgt": tgt_enc["attention_mask"],
    }

def make_dataloader(dataset, batch_size, collate_fn, num_workers=4, pin_memory=True, shuffle=True, drop_last_train=True):
    """Create a DataLoader that switches to DistributedSampler automatically under DDP."""
    if get_world_size() > 1:
        is_train = shuffle
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=is_train and drop_last_train)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=collate_fn, drop_last=is_train and drop_last_train)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          collate_fn=collate_fn, drop_last=False)
