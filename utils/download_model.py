"""
Downloads the pre-trained scaffold-gpt2-pt model from Hugging Face.

Source: https://huggingface.co/viniciusxpb/scaffold-gpt2-pt

Usage:
    python utils/download_model.py
    python utils/download_model.py --dst checkpoints
"""

import argparse
import os
import shutil

import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


REPO_ID = "viniciusxpb/scaffold-gpt2-pt"


def main():
    parser = argparse.ArgumentParser(description="Download scaffold-gpt2-pt model from Hugging Face")
    parser.add_argument("--dst", type=str, default=cfg['checkpoint']['dir'],
                        help="Destination directory (default: checkpoints)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if model already exists")
    args = parser.parse_args()

    dst = os.path.join(args.dst, "model.pt")

    if not args.force and os.path.exists(dst):
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"Model already downloaded ({size_mb:.0f} MB). Use --force to re-download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    print(f"=== Downloading Scaffold GPT-2 Model ===")
    print(f"  Repo:        {REPO_ID}")
    print(f"  Destination: {dst}")
    print()

    os.makedirs(args.dst, exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.pt",
    )

    if os.path.exists(dst):
        os.remove(dst)

    try:
        os.symlink(os.path.realpath(cached_path), dst)
    except OSError:
        shutil.copy2(cached_path, dst)

    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  Downloaded: model.pt ({size_mb:.0f} MB)")
    print(f"\nModel ready at {dst}")
    print("Next step: make generate")


if __name__ == "__main__":
    main()
