from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download

from model.tokenizer import Qwen3Tokenizer


HF_DATASET_REPO = "Skylion007/openwebtext"
LOCAL_PARQUET_DIR = Path("./data/test_converstion_data")

OUTPUT_DIR = Path("./data/openweb")
TRAIN_BIN = OUTPUT_DIR / "train.bin"
VAL_BIN = OUTPUT_DIR / "val.bin"

TRAIN_RATIO = 0.999
TEXT_COLUMNS_CANDIDATES = ["text", "content", "conversation", "messages"]


def download_parquet_folder():
    LOCAL_PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(LOCAL_PARQUET_DIR),
        allow_patterns=["*.parquet"],
    )

    parquet_files = sorted(LOCAL_PARQUET_DIR.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {LOCAL_PARQUET_DIR}")

    return parquet_files


def find_text_column(df: pd.DataFrame) -> str:
    for col in TEXT_COLUMNS_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No text-like column found. Available columns: {list(df.columns)}")


def normalize_example(value) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                parts.append(f"{role}: {content}".strip())
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items())

    return str(value)


def collect_texts(parquet_files):
    texts = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        text_col = find_text_column(df)
        for value in tqdm(df[text_col], desc=f"reading {Path(parquet_path).name}"):
            text = normalize_example(value).strip()
            if text:
                texts.append(text)
    return texts


def tokenize_texts(texts, tokenizer):
    all_ids = []
    for text in tqdm(texts, desc="tokenizing"):
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)
        all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint32)


def write_bin(path: Path, token_array: np.ndarray):
    dtype = np.uint32 if token_array.max() >= 2**16 else np.uint16
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=(len(token_array),))
    arr[:] = token_array.astype(dtype)
    arr.flush()
    print(f"wrote {path} with {len(token_array):,} tokens and dtype={dtype}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # PARQUET_FILES = [
    #     "data/test_conversion_data/plain_text/train-00000-of-00080.parquet",
    #     # "data/test_conversion_data/train-00001-of-00080.parquet",
    # ]
    parquet_files = sorted(LOCAL_PARQUET_DIR.rglob("*.parquet"))


    # print("Downloading parquet files from Hugging Face...")
    # parquet_files = download_parquet_folder()
    # print(f"Found {len(parquet_files)} parquet files")

    tokenizer = Qwen3Tokenizer()

    texts = collect_texts(parquet_files)
    token_ids = tokenize_texts(texts, tokenizer)

    split_idx = int(len(token_ids) * TRAIN_RATIO)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    write_bin(TRAIN_BIN, train_ids)
    write_bin(VAL_BIN, val_ids)

    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens:   {len(val_ids):,}")
