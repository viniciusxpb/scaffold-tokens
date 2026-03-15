# Scaffold Tokens

**Teaching language models to plan ahead by telling them how much text remains.**

Scaffold Tokens is a simple but powerful technique: prepend each word in the training data with a countdown token `<ff_N>` that tells the model exactly how many words are left until the end of the text.

```
<ff_6> O  <ff_5> presidente  <ff_4> anunciou  <ff_3> novas  <ff_2> medidas  <ff_1> econômicas  <ff_0> .
```

The model learns to use this signal for **perfect length control** and **emergent structural planning** -- starting with introductory language when `<ff>` is high and shifting to conclusions when `<ff>` approaches zero.

## Key Results

Trained GPT-2 124M from scratch on Portuguese news articles (RTX 3060, 55 minutes):

| Metric | Value |
|--------|-------|
| Length control accuracy | **100%** (53/53 exact match) |
| Final word loss | 2.08 |
| Final FF loss | 0.045 |
| Training time | 55 min / 1750 steps |
| Peak VRAM | 5.8 GB / 12 GB |

The model achieves **perfect length control** across all tested ranges (50 to 999 words), including stress tests beyond the most common training distribution. Structural planning was also detected -- conclusion words appear significantly more at the end of generated text.

## How It Works

### The Problem

Standard language models have no awareness of text length. They can't reliably generate "exactly 200 words" or plan a narrative arc because they don't know where they are relative to the end.

### The Solution

Add a countdown token before each word during training:

```
Original:  O presidente anunciou novas medidas econômicas.
Scaffold:  <ff_6> O  <ff_5> presidente  <ff_4> anunciou  <ff_3> novas  <ff_2> medidas  <ff_1> econômicas  <ff_0> .
```

- `<ff_N>` counts **words** (not BPE subword tokens)
- Each word may expand to 1-4 BPE tokens, so the actual token sequence is variable-length between `<ff>` markers
- `<ff_0>` marks the final word -- no separate EOT token needed
- Works with any BPE tokenizer (we use tiktoken GPT-2)

### Why It Works

The countdown is **deterministic** -- after `<ff_399>`, the next `<ff>` is always `<ff_398>`. The model learns this pattern in ~50 training steps (loss_ff drops to near zero almost immediately). This frees up model capacity to focus on generating good text, while the countdown provides a reliable "fuel gauge" for planning.

The model learns structural planning as an emergent behavior:
- High `<ff>` values → introductory vocabulary ("According to", "The president", proper nouns)
- Low `<ff>` values → conclusive vocabulary ("said", "announced", "completed")

## Architecture

Built on [nanoGPT 1-GPU Speedrun](https://github.com/Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN), optimized for single-GPU training:

- **Model:** GPT-2 124M (12 layers, 768 dim, 6 heads)
- **Vocab:** 51,264 tokens (50,257 BPE + 1,000 FF tokens + 7 padding)
- **Attention:** FlexAttention with progressive window (64 → 1792)
- **Optimizer:** Muon (linear layers) + AdamW (embeddings)
- **Precision:** BF16
- **Compilation:** torch.compile for maximum throughput

### Token Layout

```
IDs 0-50256:      Standard GPT-2 BPE tokens (tiktoken)
ID  50256:        EOT (end of text, separates documents in shards)
IDs 50257-51256:  <ff_0> through <ff_999> (countdown tokens)
IDs 51257-51263:  Padding (unused, for 64-alignment)
```

## Resources

| Resource | Link |
|----------|------|
| **Pre-trained model** | [viniciusxpb/scaffold-gpt2-pt](https://huggingface.co/viniciusxpb/scaffold-gpt2-pt) |
| **Dataset** | [viniciusxpb/scaffold-tokens-dataset](https://huggingface.co/datasets/viniciusxpb/scaffold-tokens-dataset) |
| **Author** | [Vinícius França](https://www.linkedin.com/in/vinicius-franca-dev/) |

## Quick Start

### Prerequisites

- NVIDIA GPU with 8+ GB VRAM (tested on RTX 3060 12GB)
- [micromamba](https://mamba.readthedocs.io/) (or conda/mamba)
- PyTorch nightly (for FlexAttention)

### Setup (Linux/Mac)

```bash
make setup
```

### Setup (Windows)

Inference: we got you bro. Training: coming soon, to a bloatware near you.

```powershell
# Run as Administrator
Set-ExecutionPolicy Bypass -Scope Process; .\setup_windows.ps1
# Then in a new terminal:
make setup
```

> **Note:** Training requires Triton (Linux only). On Windows, you can download the pre-trained model and run inference directly. For training, use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) -- think of it as a free upgrade.

### Try it now (no training needed)

```bash
make download-model             # Download pre-trained model from HF
make generate                   # Generate 100 words
make generate-prompt PROMPT="O presidente"
```

### Full Training Pipeline

```bash
# 1. Download dataset (pre-tokenized shards from Hugging Face)
make download

# 2. Validate shards
make validate

# 3. Sanity check (mini model, 30 steps)
make sanity

# 4. Train (~55 min on RTX 3060)
make train

# 5. Generate text
make generate                   # 100 words, no prompt
make generate-prompt PROMPT="O presidente"
make generate-tokens            # Show with <ff_N> tokens visible

# 6. Run test suite (53 prompts)
make test-prompts
```

## Dataset

Pre-tokenized Portuguese news articles with scaffold tokens, hosted on Hugging Face:

**[viniciusxpb/scaffold-tokens-dataset](https://huggingface.co/datasets/viniciusxpb/scaffold-tokens-dataset)**

| Field | Value |
|-------|-------|
| Source | [Folha de S.Paulo news articles](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) (public domain) |
| Language | pt-BR |
| Total tokens | ~208M |
| Format | Binary shards (uint16) |
| License | CC0 (Public Domain) |

### Data Pipeline

```
Raw news articles (public domain)
    │
    ▼
JSON files with <ff_N> annotations
    { "id": 1, "content-ff": "<ff_N> word <ff_N-1> word ..." }
    │
    ▼
Binary shards (.bin, uint16) ← downloaded from Hugging Face
    Header: 256 x int32 (magic, version, token_count)
    Body:   [ff_id, bpe_tok, bpe_tok, ff_id, bpe_tok, ...]
    │
    ▼
Training (FlexAttention + Muon + torch.compile)
    │
    ▼
Checkpoint (.pt) → Inference (forced countdown generation)
```

### Shard Format

Each `.bin` file:
- **Header:** 256 int32 values. `[0]` = magic (20240520), `[1]` = version (1), `[2]` = token count
- **Body:** uint16 tokens. Documents separated by EOT (50256)
- **Pattern within documents:** `[ff_id] [bpe_tok...] [ff_id] [bpe_tok...] [ff_id] [bpe_tok...]`

The `<ff>` token precedes each word. Since Portuguese words often split into multiple BPE subwords, the spacing between `<ff>` tokens is variable (1-4+ BPE tokens per word).

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Create env and install all dependencies |
| `make download` | Download dataset from Hugging Face |
| `make download-model` | Download pre-trained model from Hugging Face |
| `make validate` | Validate shard integrity |
| `make sanity` | Quick sanity check (mini model, 30 steps) |
| `make train` | Train GPT-2 124M (~55 min on RTX 3060) |
| `make generate` | Generate 100 words (no prompt) |
| `make generate-prompt PROMPT="..."` | Generate with prompt |
| `make generate-tokens` | Generate showing `<ff_N>` scaffold tokens |
| `make test-prompts` | Run 53 test prompts with analysis |
| `make clean` | Delete shards, checkpoints, runs, and logs |

## Configuration

Everything is centralized in `config.yaml`. Key settings:

```yaml
model:
  vocab_size: 51264
  sequence_length: 4096
  device_batch_size: 8

tokenizer:
  ff_base_id: 50257    # <ff_0> = 50257, <ff_999> = 51256
  ff_max: 999           # Max 1000 words per document

training:
  num_iterations: 1750
  cooldown_iters: 640

ff:
  weight: 5.0           # FF loss weight for weighted model (A/B test)
```

## Results in Detail

### Length Control: 100% Accuracy

All 53 test prompts generated exactly the requested number of words:

| Range | Prompts | Exact Match | Avg Error |
|-------|---------|-------------|-----------|
| Short (ff_50-99) | 15 | 15/15 | 0.0 |
| Medium (ff_100-299) | 17 | 17/17 | 0.0 |
| Long (ff_300-630) | 18 | 18/18 | 0.0 |
| Stress (ff_700-999) | 3 | 3/3 | 0.0 |

### Structural Planning

Analysis of generated text shows emergent planning behavior:
- **Beginning words** (first 10 tokens): "presidente", "federal", "Michel", "Temer" -- topic-setting vocabulary
- **Ending words** (last 10 tokens): "disse", "fazer", "governo" -- conclusive vocabulary
- Conclusion words appear significantly more at the end than the beginning

### Training Curves

- `loss_ff` drops to ~0.05 within 50 steps and stays there (countdown is trivial)
- `loss_word` decreases steadily from ~11 to ~2.08 (language modeling is the real challenge)
- Total training: 1750 steps, 55 minutes, single RTX 3060

## Future Research

- **Ghost Tokens:** Post-training embedding expansion -- train with extra dormant dimensions, then unlock them to discover new semantic features without retraining the full model
- **Smart Attention:** Topic tokens (most frequent content words) prepended to documents, visible to attention but masked from loss -- giving the model a "spoiler" of what the text is about
- **Avoid Topics:** Negative conditioning tokens that teach the model to discuss a topic while avoiding specific words, enabling controlled paraphrasing
- **Rich Dataset Theory:** The hypothesis that structured dataset annotations (countdown, topics, avoidance signals) provide more training signal per token than raw text scaling

## Hardware Notes

Tested on:
- **GPU:** NVIDIA RTX 3060 12GB (SM86)
- **Precision:** BF16 (FP8 requires SM89+, not supported)
- **CPU:** Limited to 4 cores (system stability)
- **Throughput:** ~1912 ms/step, ~33k tokens/step

## License

MIT

## Author

[Vinícius França](https://www.linkedin.com/in/vinicius-franca-dev/)

## Acknowledgments

- [nanoGPT 1-GPU Speedrun](https://github.com/Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN) by Deveraux Parker -- base training architecture
- [Folha de S.Paulo News Dataset](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) by Marlesson -- Portuguese news articles
- [tiktoken](https://github.com/openai/tiktoken) -- BPE tokenization
