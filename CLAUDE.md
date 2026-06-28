# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanoGPT is a minimalist GPT training/finetuning/sampling codebase. `train.py` (~300 lines) is the training loop, `model.py` (~300 lines) is the GPT model definition. The code is designed to be simple, hackable, and reproduce GPT-2 (124M) on OpenWebText.

## Configuration system

`configurator.py` is an unconventional config system: it's `exec()`-ed at module level in `train.py`, `sample.py`, and `bench.py` to override global variables. Config files are plain Python files that assign to globals; CLI args use `--key=value` syntax. Do not `import configurator` — the scripts literally `exec(open('configurator.py').read())`.

## Key commands

**Data preparation:**
- Character-level Shakespeare: `python data/shakespeare_char/prepare.py`
- BPE-tokenized Shakespeare: `python data/shakespeare/prepare.py`
- OpenWebText (GPT-2 BPE): `python data/openwebtext/prepare.py`

**Training:**
- Baby GPT on Shakespeare (single GPU): `python train.py config/train_shakespeare_char.py`
- On CPU/Macbook: add `--device=cpu --compile=False`
- On Apple Silicon: use `--device=mps`
- GPT-2 124M reproduction (8 GPUs): `torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py`
- Finetuning GPT-2 on Shakespeare: `python train.py config/finetune_shakespeare.py`

**Inference:**
- Sample from your own checkpoint: `python sample.py --out_dir=<out_dir>`
- Sample from pretrained GPT-2: `python sample.py --init_from=gpt2-xl --start="Your prompt"`

**Evaluation baselines:** `python train.py config/eval_gpt2.py` (and `eval_gpt2_medium.py`, `eval_gpt2_large.py`, `eval_gpt2_xl.py`)

**Benchmarking:** `python bench.py`

## Architecture

### model.py — GPT model

- `GPTConfig` dataclass: `block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- `GPT(nn.Module)`: full transformer with `wte` (token embedding), `wpe` (position embedding), `h` (list of `Block`s), `ln_f` (final LayerNorm), and `lm_head` (output projection). Uses **weight tying** between `wte` and `lm_head`.
- `Block`: pre-LayerNorm transformer block → `LayerNorm` → `CausalSelfAttention` → residual, then `LayerNorm` → `MLP` → residual.
- `CausalSelfAttention`: Uses PyTorch's `scaled_dot_product_attention` (flash attention) when available, falls back to manual causal attention with a lower-triangular mask buffer.
- `MLP`: 4x expansion factor with GELU activation.
- `GPT.from_pretrained()`: Loads OpenAI GPT-2 checkpoints via HuggingFace `transformers`, transposing Conv1D weights to Linear.
- `GPT.configure_optimizers()`: AdamW with weight decay applied only to 2D+ parameters (Linear weights, embeddings). Biases and LayerNorms get no weight decay.
- `GPT.crop_block_size()`: Surgical reduction of context length for loaded checkpoints.
- `GPT.estimate_mfu()`: Model FLOPS utilization relative to A100 bfloat16 peak (312 TFLOPS).
- `LayerNorm`: Custom LayerNorm with optional bias (PyTorch's built-in doesn't support `bias=False`).

### train.py — Training loop

- Supports single GPU, DDP (`torchrun`), and CPU/MPS.
- Data: numpy `memmap` of raw uint16 token ids from `.bin` files. Batches are random contiguous chunks of `block_size` tokens.
- Mixed precision: `bfloat16` by default, `float16` with GradScaler, or `float32`.
- LR schedule: linear warmup for `warmup_iters` steps, then cosine decay to `min_lr`.
- Gradient accumulation with `gradient_accumulation_steps` micro-batches.
- `init_from` modes: `'scratch'`, `'resume'` (from `out_dir/ckpt.pt`), or `'gpt2'`/`'gpt2-medium'`/`'gpt2-large'`/`'gpt2-xl'`.

### sample.py — Inference

- Loads checkpoint from `out_dir/ckpt.pt` or HuggingFace GPT-2 weights.
- Uses tiktoken GPT-2 encoding by default, or character-level encoding from `meta.pkl` if available.
- Generation with temperature and top-k sampling via `model.generate()`.

### Data format

- Tokenized datasets are stored as flat uint16 `.bin` files (`train.bin`, `val.bin`).
- `meta.pkl` (optional) contains `vocab_size`, `stoi`/`itos` for character-level datasets. BPE datasets omit this and rely on tiktoken.
- `data/shakespeare_char/` — character-level encoding (vocab size 65).
- `data/shakespeare/` — GPT-2 BPE encoding.
- `data/openwebtext/` — GPT-2 BPE encoding (~9B tokens).
