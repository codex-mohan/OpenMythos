# Architectural Improvements Over the Original OpenMythos

This document describes the critical architectural flaws in the original design
and how they were fixed.  Every change is backward-compatible at the API level
and covered by the existing test suite.

---

## 1. Multi-Layer Recurrent Block (was: single-layer loop)

### The Problem

The original `RecurrentBlock` contained **one single `TransformerBlock`** and
looped it `max_loop_iters` times:

```
for t in range(n_loops):
    h = one_attention(h) + one_moe_ffn(h) + injection
```

This meant the model had **one set of attention heads and one FFN for ALL
computation across all loop iterations**.  The only things that changed across
loops were:
- A weak sinusoidal loop-index embedding on `dim // 8` channels
- A tiny LoRA adapter (rank 8)
- The evolving hidden state `h`

One attention pattern cannot simultaneously handle token-level syntax,
cross-paragraph entity resolution, and high-level reasoning.  The effective
depth was `1 × max_loop_iters` — equivalent to an RNN unrolled with weight
tying, not a deep transformer.

### The Fix

The recurrent block now contains **`recurrent_layers` distinct
`TransformerBlock`s** that compose the recurrent stack.  The full stack is
looped `max_loop_iters` times:

```
for t in range(n_loops):
    h = loop_index_embedding(h, t)
    combined = norm(h + e)                 # inject frozen prelude output
    for block in self.blocks:              # N distinct blocks, each pass
        combined = block(combined)         #   = attention + shared MoE FFN
    h = injection(h, e, combined)          # LTI-stable update
    h_out += act_weight * h                # ACT-weighted accumulation
```

**Key design decisions:**

- **Distinct attention per layer.**  Each block has its own MLA/GQA weights
  (`q_down`, `q_up_nope`, `q_up_rope`, `kv_down`, `kv_up`, `wo`).  Different
  layers learn different attention patterns.

- **Shared MoE across layers.**  All recurrent blocks reference the same
  `MoEFFN` instance (passed via `shared_ffn`).  Different layers route to
  different experts because the hidden state feeding the router differs at
  each layer.  This keeps parameter count manageable while preserving domain
  breadth.

- **Effective depth = `recurrent_layers × max_loop_iters`.**  The 3B variant
  went from 16 effective depth (1×16) to 36 (12×3) with minimal parameter
  increase — 12× MLA attention weights (~11M extra) vs. the 2.7B MoE that is
  shared.

| Variant | Old (1×N) | New (L×T) | Effective depth |
|---|---|---|---|
| 200M | — | 8×4 | 32 |
| 1B | 1×16 | 8×2 | 16 |
| 3B | 1×16 | 12×3 | 36 |
| 10B | 1×24 | 16×4 | 64 |
| 50B | 1×32 | 20×4 | 80 |
| 100B | 1×32 | 24×4 | 96 |
| 500B | 1×48 | 32×5 | 160 |
| 1T | 1×64 | 40×6 | 240 |

### Optional refinements for the future

- **Staged loop counts.**  Not all recurrent layers need to run the same
  number of times.  Early layers could loop fewer times than late layers,
  creating a "funnel" of compute depth.

- **Per-layer LTI injection.**  Currently injection happens once per outer
  loop.  Injecting the frozen input `e` after every layer (not just every
  loop) may improve stability at extreme depths.

---

## 2. Vocabulary Management (was: hardcoded `vocab_size=32000`)

### The Problem

Every variant and the config default hardcoded `vocab_size=32000`.  Only the
training script knew to override it via `cfg.vocab_size = encoding.vocab_size`.
This meant:

- Using a different tokenizer silently produced a model with the wrong
  embedding matrix size.
- No visible connection between tokenizer choice and model config.
- No support for custom-trained tokenizers.

### The Fix

Three additions to `open_mythos.tokenizer` and `MythosConfig`:

#### `MythosConfig.with_vocab(model_id | int)`

```python
# From HuggingFace model ID — downloads tokenizer config only (KB, not weights)
cfg = mythos_3b().with_vocab("openai/gpt-oss-20b")    # → 199,998
cfg = mythos_3b().with_vocab("Qwen/Qwen3-0.5B")       # → 151,936
# Or explicit integer
cfg = mythos_200m().with_vocab(50000)
```

Returns a new config with `vocab_size` set correctly.  Does **not** download
model weights — only the tokenizer JSON files.

#### `get_vocab_size(model_id) -> int`

Quick lookup without building a tokenizer instance:

```python
from open_mythos import get_vocab_size
get_vocab_size("openai/gpt-oss-20b")  # → 199,998
```

Loads (and caches) the tokenizer config.  Subsequent calls are instant.

#### `train_bpe_tokenizer(texts, vocab_size, output_dir)`

Train a custom BPE tokenizer from raw text:

```python
from open_mythos import train_bpe_tokenizer

ds = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True, split="train")
texts = (x["text"] for x in ds.take(100_000))
tokenizer = train_bpe_tokenizer(texts, vocab_size=50000, output_dir="my_tok")
```

Clones the pre-tokenizer and normalizer from the base tokenizer
(`openai/gpt-oss-20b` by default) and trains a new BPE vocabulary on your data.
The result is saved as a HuggingFace-compatible tokenizer.

#### `MythosTokenizer` now accepts pre-built tokenizers

```python
tok = MythosTokenizer()                          # default: gpt-oss-20b
tok = MythosTokenizer("meta-llama/Llama-3.2-1B") # any HF model
tok = MythosTokenizer("./my_tokenizer")          # local path
tok = MythosTokenizer(custom_bpe)                # pre-built instance
```

---

## 3. 200M-Parameter Variant

A new small variant comparable in scale to GPT-2 Small (124M) but with MoE
breadth and recurrent depth:

```python
from open_mythos import mythos_200m

cfg = mythos_200m().with_vocab("openai/gpt-oss-20b")
model = OpenMythos(cfg)
```

| Property | Value |
|---|---|
| dim | 1024 |
| recurrent layers | 8 |
| loops | 4 |
| effective depth | 32 |
| experts | 30 routed + 2 shared |
| prelude / coda | 4 blocks each |
| total params | 203M |
| activated/token | ~40M |
| fit on | single 8-GPU node |

---

## 4. FSDP Compatibility Fix

The original training script used `ModuleWrapPolicy({TransformerBlock,
RecurrentBlock})`.  With the shared MoE (one `MoEFFN` referenced by multiple
`TransformerBlock`s), this would cause FSDP to try to register the same
parameters in multiple sharded units — a fatal error.

**Fix:** wrap only `RecurrentBlock`.  The shared MoE lives inside a single
FSDP unit, avoiding duplicate parameter registration.  Prelude/coda blocks
(which have their own dense FFNs, not shared) live inside the root FSDP
wrapper.
