"""
Downloads pre-tokenized scaffold token shards from Hugging Face.

The dataset contains Portuguese news articles with <ff_N> countdown tokens,
already converted to binary shards (uint16) ready for training.

Source: https://huggingface.co/datasets/viniciusxpb/scaffold-tokens-dataset

Usage:
    python utils/download_dataset.py
    python utils/download_dataset.py --dst data/shards
"""

import argparse
import os
import shutil

import numpy as np
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


REPO_ID = "viniciusxpb/scaffold-tokens-dataset"


def is_already_downloaded(dst):
    """Check if shards already exist and are valid."""
    for split in ["train", "val"]:
        split_dir = os.path.join(dst, split)
        if not os.path.isdir(split_dir):
            return False
        bins = [f for f in os.listdir(split_dir) if f.endswith(".bin")]
        if not bins:
            return False
        for fname in bins:
            path = os.path.join(split_dir, fname)
            try:
                header = np.fromfile(path, dtype=np.int32, count=3)
                if header[0] != cfg['shards']['magic']:
                    return False
            except Exception:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download scaffold-tokens dataset from Hugging Face")
    parser.add_argument("--dst", type=str, default=cfg['data']['shards_dir'],
                        help="Destination directory for shards (default: data/shards)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if shards already exist")
    args = parser.parse_args()

    if not args.force and is_already_downloaded(args.dst):
        print("Dataset already downloaded. Use --force to re-download.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    print(f"=== Downloading Scaffold Tokens Dataset ===")
    print(f"  Repo:        {REPO_ID}")
    print(f"  Destination: {args.dst}/")
    print()

    # Download all .bin files from the repo
    snapshot_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns="data/**/*.bin",
    )

    # Link shard files to destination
    src_data = os.path.join(snapshot_dir, "data")
    for split in ["train", "val"]:
        src_split = os.path.join(src_data, split)
        dst_split = os.path.join(args.dst, split)
        if not os.path.isdir(src_split):
            continue
        os.makedirs(dst_split, exist_ok=True)

        for fname in sorted(os.listdir(src_split)):
            if not fname.endswith(".bin"):
                continue
            src = os.path.join(src_split, fname)
            dst = os.path.join(dst_split, fname)
            if os.path.exists(dst):
                os.remove(dst)
            # Symlink saves disk space; copy as fallback
            try:
                os.symlink(os.path.realpath(src), dst)
            except OSError:
                shutil.copy2(src, dst)

    # Verify
    print("\n=== Verifying downloads ===")
    total_tokens = 0
    for split in ["train", "val"]:
        split_dir = os.path.join(args.dst, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".bin"):
                continue
            path = os.path.join(split_dir, fname)
            header = np.fromfile(path, dtype=np.int32, count=3)
            assert header[0] == cfg['shards']['magic'], f"Invalid magic number in {path}"
            tokens = header[2]
            total_tokens += tokens
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {split}/{fname}: {tokens:,} tokens ({size_mb:.0f} MB)")

    print(f"\n  Total: {total_tokens:,} tokens")
    print(f"\nDataset ready at {args.dst}/")
    print("Next step: make validate")


if __name__ == "__main__":
    main()
