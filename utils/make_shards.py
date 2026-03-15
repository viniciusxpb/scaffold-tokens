"""
Converts JSONs from json_content_ff into binary .bin shards
compatible with the modded-nanogpt data loader.

Format of each shard:
  Header: 256 x int32 (magic=20240520, version=1, token_count)
  Body:   uint16 tokens

Each document becomes: ff_id bpe_tok... ff_id bpe_tok... EOT
(ff counts words, each word can be 1+ BPE tokens)

Streaming: does not accumulate everything in RAM. Writes shards incrementally.

Usage:
    python utils/make_shards.py [--limit N] [--workers N] [--val-split 0.05]
"""

import json
import os
import re
import argparse
import random
import numpy as np
import tiktoken
from multiprocessing import Pool
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MAX_WORKERS = cfg['hardware']['max_workers']
FF_BASE_ID = cfg['tokenizer']['ff_base_id']
EOT_ID = cfg['tokenizer']['eot_id']
SHARD_SIZE = cfg['shards']['shard_size']
MAGIC = cfg['shards']['magic']
VERSION = cfg['shards']['version']

DEFAULT_SRC = os.path.join("data", "json_content_ff")
DEFAULT_DST = cfg['data']['shards_dir']


def tokenize_doc(fpath: str) -> list[int] | None:
    """Converts a content-ff JSON into a sequence of token IDs.
    Returns None if the doc has more than 1000 words (ff > 999)."""
    enc = tiktoken.get_encoding("gpt2")

    with open(fpath, "r", encoding="utf-8") as f:
        doc = json.load(f)

    text = doc["content-ff"]
    pairs = re.findall(r"<ff_(\d+)>\s*([^<]*)", text)

    # Filter empty words and recalculate countdown
    words = [w.strip() for _, w in pairs]
    words = [w for w in words if w]

    if len(words) > cfg['tokenizer']['ff_max'] + 1:
        return None

    tokens = []
    n = len(words)
    for i, word in enumerate(words):
        ff_id = FF_BASE_ID + (n - 1 - i)
        bpe = enc.encode(word)
        tokens.append(ff_id)
        tokens.extend(bpe)

    tokens.append(EOT_ID)
    return tokens


class ShardWriter:
    """Writes shards incrementally without accumulating everything in RAM."""

    def __init__(self, out_dir: str, prefix: str = "shard"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix
        self.buffer = []
        self.shard_idx = 0
        self.total_tokens = 0

    def add(self, tokens: list[int]) -> None:
        self.buffer.extend(tokens)
        while len(self.buffer) >= SHARD_SIZE:
            self._flush(SHARD_SIZE)

    def close(self) -> None:
        if self.buffer:
            self._flush(len(self.buffer))

    def _flush(self, n: int) -> None:
        chunk = self.buffer[:n]
        self.buffer = self.buffer[n:]

        header = np.zeros(256, dtype=np.int32)
        header[0] = MAGIC
        header[1] = VERSION
        header[2] = len(chunk)

        body = np.array(chunk, dtype=np.uint16)
        path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard_idx:03d}.bin")

        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(body.tobytes())

        self.total_tokens += len(chunk)
        self.shard_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--val-split", type=float, default=cfg['shards']['val_split'])
    parser.add_argument("--seed", type=int, default=cfg['shards']['seed'])
    parser.add_argument("--src", type=str, default=DEFAULT_SRC, help="Directory with JSONs")
    parser.add_argument("--dst", type=str, default=DEFAULT_DST, help="Output directory for shards")
    args = parser.parse_args()
    args.workers = min(args.workers, MAX_WORKERS)

    SRC_DIR = args.src
    DST_DIR = args.dst

    files = sorted(
        os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if f.endswith(".json")
    )
    if args.limit > 0:
        files = files[: args.limit]

    # Shuffle before splitting train/val
    random.seed(args.seed)
    random.shuffle(files)

    n_val = max(1, int(len(files) * args.val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]

    train_dir = os.path.join(DST_DIR, "train")
    val_dir = os.path.join(DST_DIR, "val")

    for split_name, split_files, out_dir in [
        ("train", train_files, train_dir),
        ("val", val_files, val_dir),
    ]:
        print(f"[{split_name}] Tokenizing {len(split_files)} files with {args.workers} workers...")
        writer = ShardWriter(out_dir)

        skipped = 0
        with Pool(args.workers) as pool:
            for i, toks in enumerate(pool.imap(tokenize_doc, split_files, chunksize=64), 1):
                if toks is None:
                    skipped += 1
                    continue
                writer.add(toks)
                if i % 10000 == 0:
                    print(f"  {i}/{len(split_files)} ({writer.total_tokens:,} tokens, {writer.shard_idx} shards, {skipped} discarded)")

        writer.close()
        print(f"  Done: {writer.total_tokens:,} tokens in {writer.shard_idx} shards -> {out_dir}/ ({skipped} docs discarded for ff>999)")

    print("Done!")


if __name__ == "__main__":
    main()
