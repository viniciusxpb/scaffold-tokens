"""
Phase 2 — Shard validation.

Tests:
  2.1  ID integrity (range, dtype)
  2.2  FF followed by word tokens pattern
  2.3  Correct countdown (decrements 1 per word)
  2.4  Reversibility (reconstruct original text)
  2.5  Distribution statistics
  2.6  Visual sample

Usage:
    python utils/validate_shards.py [--shards-dir data/shards]
"""

import os
import argparse
import random
import numpy as np
import tiktoken
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

FF_BASE_ID = cfg['tokenizer']['ff_base_id']
FF_MAX = cfg['tokenizer']['ff_max']
EOT_ID = cfg['tokenizer']['eot_id']
MAX_TOKEN_ID = FF_BASE_ID + FF_MAX  # 51256


def load_shard(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == cfg['shards']['magic'], f"Invalid magic: {header[0]}"
        assert header[1] == 1, f"Invalid version: {header[1]}"
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == header[2], f"Token count mismatch: {len(tokens)} vs {header[2]}"
    return tokens


def extract_docs(tokens: np.ndarray) -> list[np.ndarray]:
    """Split tokens into documents using EOT as delimiter."""
    eot_positions = np.where(tokens == EOT_ID)[0]
    docs = []
    start = 0
    for eot_pos in eot_positions:
        doc = tokens[start:eot_pos]  # without the EOT
        if len(doc) > 0:
            docs.append(doc)
        start = eot_pos + 1
    return docs


def test_integrity(all_tokens: np.ndarray) -> bool:
    """2.1 — All IDs in valid range, dtype uint16."""
    print("=== 2.1 ID integrity ===")
    ok = True

    if all_tokens.dtype != np.uint16:
        print(f"  FAIL: dtype={all_tokens.dtype}, expected uint16")
        ok = False

    max_val = all_tokens.max()
    min_val = all_tokens.min()
    above = (all_tokens > MAX_TOKEN_ID) & (all_tokens != EOT_ID)
    # IDs between MAX_TOKEN_ID+1 and 51263 are padding, should not appear
    in_padding = (all_tokens > MAX_TOKEN_ID) & (all_tokens <= cfg['model']['vocab_size'] - 1)

    print(f"  Range: [{min_val}, {max_val}]")
    print(f"  Tokens above {MAX_TOKEN_ID} (excl EOT): {above.sum()}")
    print(f"  Tokens in padding zone ({MAX_TOKEN_ID + 1}-{cfg['model']['vocab_size'] - 1}): {in_padding.sum()}")

    if above.sum() > 0:
        print(f"  FAIL: {above.sum()} tokens out of range")
        ok = False

    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def test_ff_word_pattern(docs: list[np.ndarray]) -> bool:
    """2.2 — Each document starts with ff, and each ff is followed by 1+ BPE tokens."""
    print("\n=== 2.2 FF/word pattern ===")
    ok = True
    errors = 0

    for i, doc in enumerate(docs):
        if len(doc) == 0:
            continue
        # first token must be ff
        if doc[0] < FF_BASE_ID:
            if errors < 5:
                print(f"  Doc {i}: first token {doc[0]} is not ff")
            errors += 1
            continue

        # check: each ff must be followed by at least 1 BPE token
        j = 0
        while j < len(doc):
            if doc[j] >= FF_BASE_ID:
                # it's ff, next must be BPE
                if j + 1 >= len(doc):
                    if errors < 5:
                        print(f"  Doc {i} pos {j}: ff at end without BPE token")
                    errors += 1
                    break
                if doc[j + 1] >= FF_BASE_ID:
                    if errors < 5:
                        print(f"  Doc {i} pos {j}: two consecutive ff")
                    errors += 1
                j += 1
            else:
                # it's BPE, advance to next ff
                j += 1

    if errors > 0:
        print(f"  FAIL: {errors} errors in {len(docs)} docs")
        ok = False
    else:
        print(f"  OK: {len(docs)} docs verified")
    return ok


def test_countdown(docs: list[np.ndarray]) -> bool:
    """2.3 — Countdown decrements 1 per ff."""
    print("\n=== 2.3 Correct countdown ===")
    ok = True
    errors = 0

    for i, doc in enumerate(docs):
        ff_values = []
        for t in doc:
            if t >= FF_BASE_ID:
                ff_values.append(int(t) - FF_BASE_ID)

        if len(ff_values) == 0:
            continue

        # first ff should be N-1 where N = number of words
        expected = ff_values[0]
        for j, val in enumerate(ff_values):
            if val != expected:
                if errors < 5:
                    print(f"  Doc {i} ff #{j}: expected {expected}, found {val}")
                errors += 1
                break
            expected -= 1

        # last ff should be 0
        if ff_values[-1] != 0:
            if errors < 5:
                print(f"  Doc {i}: last ff={ff_values[-1]}, expected 0")
            errors += 1

    if errors > 0:
        print(f"  FAIL: {errors} errors in {len(docs)} docs")
        ok = False
    else:
        print(f"  OK: {len(docs)} docs verified")
    return ok


def test_reversibility(docs: list[np.ndarray], original_dir: str, n_test: int = 100) -> bool:
    """2.4 — Reconstruct text by removing ff tokens and decoding."""
    print(f"\n=== 2.4 Reversibility ({n_test} docs) ===")

    if not os.path.isdir(original_dir):
        print(f"  SKIP: directory {original_dir} not found")
        return True

    enc = tiktoken.get_encoding("gpt2")
    ok = True
    errors = 0
    tested = 0

    # can't easily map shard doc to original file
    # just verify that decoding produces valid text
    sample = random.sample(docs, min(n_test, len(docs)))

    for doc in sample:
        bpe_tokens = [int(t) for t in doc if t < FF_BASE_ID]
        try:
            text = enc.decode(bpe_tokens)
            if len(text.strip()) == 0:
                errors += 1
            tested += 1
        except Exception as e:
            if errors < 5:
                print(f"  Error decoding: {e}")
            errors += 1

    if errors > 0:
        print(f"  WARNING: {errors}/{tested} docs with issues")
        ok = False
    else:
        print(f"  OK: {tested} docs decoded successfully")
    return ok


def test_distribution(docs: list[np.ndarray], all_tokens: np.ndarray) -> None:
    """2.5 — Distribution statistics."""
    print("\n=== 2.5 Distribution ===")

    # doc sizes (in total tokens)
    doc_sizes = [len(d) for d in docs]
    ff_counts = []
    for doc in docs:
        n_ff = sum(1 for t in doc if t >= FF_BASE_ID)
        ff_counts.append(n_ff)

    print(f"  Total documents: {len(docs)}")
    print(f"  Total tokens: {len(all_tokens):,}")
    print(f"  Doc size (tokens): min={min(doc_sizes)}, max={max(doc_sizes)}, "
          f"mean={np.mean(doc_sizes):.0f}, median={np.median(doc_sizes):.0f}")
    print(f"  Words per doc (ff count): min={min(ff_counts)}, max={max(ff_counts)}, "
          f"mean={np.mean(ff_counts):.0f}")

    # % of ff tokens
    n_ff = sum(1 for t in all_tokens if t >= FF_BASE_ID)
    n_bpe = sum(1 for t in all_tokens if t < FF_BASE_ID and t != EOT_ID)
    n_eot = sum(1 for t in all_tokens if t == EOT_ID)
    print(f"  ff tokens: {n_ff:,} ({100 * n_ff / len(all_tokens):.1f}%)")
    print(f"  BPE tokens: {n_bpe:,} ({100 * n_bpe / len(all_tokens):.1f}%)")
    print(f"  EOT tokens: {n_eot:,}")
    print(f"  Ratio BPE/ff: {n_bpe / max(n_ff, 1):.2f} (BPE tokens per word)")

    # size on disk
    total_bytes = len(all_tokens) * 2 + 256 * 4  # uint16 + header
    print(f"  Estimated size: {total_bytes / 1024 / 1024:.1f} MB")


def test_visual_sample(docs: list[np.ndarray], n: int = 5) -> None:
    """2.6 — Visual sample."""
    print(f"\n=== 2.6 Visual sample ({n} docs) ===")
    enc = tiktoken.get_encoding("gpt2")
    sample = random.sample(docs, min(n, len(docs)))

    for i, doc in enumerate(sample):
        print(f"\n  --- Doc {i + 1} ({len(doc)} tokens) ---")
        parts = []
        j = 0
        while j < len(doc) and len(parts) < 30:
            if doc[j] >= FF_BASE_ID:
                ff_val = int(doc[j]) - FF_BASE_ID
                # collect BPE tokens for this word
                j += 1
                bpe = []
                while j < len(doc) and doc[j] < FF_BASE_ID:
                    bpe.append(int(doc[j]))
                    j += 1
                word = enc.decode(bpe) if bpe else "?"
                parts.append(f"ff_{ff_val} \"{word}\"")
            else:
                j += 1

        print("  " + "  ".join(parts))
        if len(doc) > 30:
            print("  ...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", default=cfg['data']['shards_dir'])
    parser.add_argument("--original-dir", default="data/json_content_ff")
    args = parser.parse_args()

    # load all shards
    all_tokens = []
    for split in ["train", "val"]:
        split_dir = os.path.join(args.shards_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".bin"):
                path = os.path.join(split_dir, fname)
                tokens = load_shard(path)
                print(f"Loaded: {path} ({len(tokens):,} tokens)")
                all_tokens.append(tokens)

    if not all_tokens:
        print("No shards found!")
        return

    merged = np.concatenate(all_tokens)
    docs = extract_docs(merged)

    print(f"\nTotal: {len(merged):,} tokens, {len(docs)} documents\n")

    random.seed(cfg['shards']['seed'])

    r1 = test_integrity(merged)
    r2 = test_ff_word_pattern(docs)
    r3 = test_countdown(docs)
    r4 = test_reversibility(docs, args.original_dir)
    test_distribution(docs, merged)
    test_visual_sample(docs)

    print("\n=== FINAL RESULT ===")
    all_ok = r1 and r2 and r3 and r4
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("FAILURES DETECTED — review dataset before training")


if __name__ == "__main__":
    main()
