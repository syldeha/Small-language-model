from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.tokenizer import Qwen3Tokenizer


PARQUET_FILES = [
    r"data\test_conversion_data\train-00000-of-00080.parquet",
    r"data\test_conversion_data\train-00001-of-00080.parquet",
]


OUTPUT_DIR = Path(r"F:\COURS IA\4A\PREPA INTERVIEW\AI CODING\AI\LLM projet\data\openweb")
TRAIN_BIN = OUTPUT_DIR / "train.bin"
VAL_BIN = OUTPUT_DIR / "val.bin"

TRAIN_RATIO = 0.999  # keep val small, like nanochat style
TEXT_COLUMNS_CANDIDATES = ["text", "content", "conversation", "messages"]


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
    if token_array.max() >= 2**16:
        dtype = np.uint32
    else:
        dtype = np.uint16

    arr = np.memmap(path, dtype=dtype, mode="w+", shape=(len(token_array),))
    arr[:] = token_array.astype(dtype)
    arr.flush()
    print(f"wrote {path} with {len(token_array):,} tokens and dtype={dtype}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Qwen3Tokenizer()

    texts = collect_texts(PARQUET_FILES)
    token_ids = tokenize_texts(texts, tokenizer)

    split_idx = int(len(token_ids) * TRAIN_RATIO)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    write_bin(TRAIN_BIN, train_ids)
    write_bin(VAL_BIN, val_ids)

    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens:   {len(val_ids):,}")
