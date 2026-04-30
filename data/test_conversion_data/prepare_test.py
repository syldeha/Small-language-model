from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from model.tokenizer import Qwen3Tokenizer


DATASET_NAME = "Skylion007/openwebtext"
OUTPUT_DIR = Path("./data/openweb")
CACHE_DIR = Path("./data/hf_cache")

TRAIN_BIN = OUTPUT_DIR / "train.bin"
VAL_BIN = OUTPUT_DIR / "val.bin"

VAL_RATIO = 0.001
SEED = 2357
NUM_PROC = 8
TOTAL_BATCHES_TO_WRITE = 1024


def process(example):
    ids = TOKENIZER.encode(example["text"])
    ids.append(TOKENIZER.eos_token_id)
    return {"ids": ids, "len": len(ids)}


def choose_dtype(tokenized_split):
    max_token_id = 0
    for ids in tokenized_split["ids"]:
        if ids:
            local_max = max(ids)
            if local_max > max_token_id:
                max_token_id = local_max
    return np.uint16 if max_token_id < 2**16 else np.uint32


def write_split_to_bin(dset, out_path: Path):
    arr_len = np.sum(dset["len"], dtype=np.uint64)
    if arr_len == 0:
        raise ValueError(f"{out_path.name}: tokenized split is empty")

    dtype = choose_dtype(dset)
    arr = np.memmap(out_path, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    for batch_idx in tqdm(range(TOTAL_BATCHES_TO_WRITE), desc=f"writing {out_path.name}"):
        shard = dset.shard(
            num_shards=TOTAL_BATCHES_TO_WRITE,
            index=batch_idx,
            contiguous=True,
        ).with_format("numpy")

        if len(shard) == 0:
            continue

        arr_batch = np.concatenate(shard["ids"])
        arr[idx: idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()
    print(f"wrote {out_path} with {arr_len:,} tokens and dtype={dtype}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    TOKENIZER = Qwen3Tokenizer()

    dataset = load_dataset(
        DATASET_NAME,
        cache_dir=str(CACHE_DIR),
    )

    split_dataset = dataset["train"].train_test_split(
        test_size=VAL_RATIO,
        seed=SEED,
        shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    write_split_to_bin(tokenized["train"], TRAIN_BIN)
    write_split_to_bin(tokenized["val"], VAL_BIN)

    print(f"train rows: {len(tokenized['train']):,}")
    print(f"val rows:   {len(tokenized['val']):,}")
