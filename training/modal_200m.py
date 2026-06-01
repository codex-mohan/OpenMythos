"""
Train a 200M-parameter OpenMythos model on FineWeb-Edu using Modal.com.

Single H100 GPU.  Targets 1 epoch of sample-10BT (~10B tokens).
Checkpoints saved to a Modal Volume every 500 steps.

ACT halting is DISABLED during pretraining (use_act=False) — the model
runs all loops and returns the final hidden state.  ACT will be enabled
during later RL fine-tuning when per-task difficulty signal is available.

Usage:
    pip install modal
    modal setup
    modal run training/modal_200m.py

Estimated cost: ~$25-40 for a full 10B-token run on H100.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App("mythos-200m-train")

GPU = "H100"
GPU_COUNT = 1
TRAIN_TIMEOUT = 24 * 60 * 60  # 24 hours

CHECKPOINT_VOLUME = modal.Volume.from_name(
    "mythos-200m-checkpoints", create_if_missing=True
)
CHECKPOINT_DIR = Path("/checkpoints")

# ---------------------------------------------------------------------------
# Container image — PyTorch + HuggingFace + open-mythos
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "loguru>=0.7.3",
        "sentencepiece",
        "tiktoken",
    )
    .pip_install(  # Flash Attention 2 for GQA path (MLA uses manual SDPA)
        "flash-attn>=2.8.3", extra_options="--no-build-isolation"
    )
    # Install open-mythos from the local source tree
    .add_local_dir(
        Path(__file__).parent.parent,
        remote_path="/root/open_mythos_src",
    )
    .run_commands(
        "cd /root/open_mythos_src && pip install -e . --no-deps",
    )
)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=f"{GPU}:{GPU_COUNT}",
    volumes={CHECKPOINT_DIR: CHECKPOINT_VOLUME},
    timeout=TRAIN_TIMEOUT,
)
def train():
    import torch
    import torch.nn as nn
    from loguru import logger
    from datasets import load_dataset

    from open_mythos import OpenMythos, mythos_200m, MythosTokenizer

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = mythos_200m().with_vocab("openai/gpt-oss-20b")
    cfg.max_seq_len = 2048
    cfg.max_loop_iters = 4
    cfg.dropout = 0.0
    cfg.use_act = False  # disable ACT during pretraining

    seq_len = cfg.max_seq_len
    micro_batch = 4
    grad_accum = 16  # effective batch = 4 × 16 × 2048 = 131K tokens/step
    total_tokens_target = 10_000_000_000  # 1 epoch of sample-10BT
    lr = 3e-4
    min_lr = 3e-5
    weight_decay = 0.1
    warmup_steps = 500
    ckpt_every = 500
    log_every = 10
    ckpt_dir = CHECKPOINT_DIR / "mythos_200m"

    device = torch.device("cuda")
    bf16_ok = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_amp = device.type == "cuda"

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    assert vocab_size == cfg.vocab_size, (
        f"Tokenizer vocab {vocab_size} != config vocab {cfg.vocab_size}"
    )
    logger.info(f"Tokenizer: {encoding.model_id} | vocab_size={vocab_size:,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Building model...")
    model = OpenMythos(cfg).to(device=device, dtype=amp_dtype if use_amp else torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {n_params:,} total params | {n_trainable:,} trainable | "
        f"effective depth = {cfg.recurrent_layers}×{cfg.max_loop_iters} = "
        f"{cfg.recurrent_layers * cfg.max_loop_iters}"
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True if device.type == "cuda" else False,
    )

    tokens_per_step = micro_batch * grad_accum * seq_len
    total_steps = total_tokens_target // tokens_per_step

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    logger.info(
        f"Training: {total_steps:,} steps | {tokens_per_step:,} tok/step | "
        f"{total_tokens_target / 1e9:.1f}B tokens target | "
        f"batch={micro_batch}×{grad_accum} acc | seq_len={seq_len}"
    )

    # ------------------------------------------------------------------
    # Dataset — streaming FineWeb-Edu
    # ------------------------------------------------------------------
    logger.info("Loading dataset (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    def token_generator():
        """Yield tokenised chunks of `seq_len + 1` for input/target shifting."""
        buf: list[int] = []
        for sample in ds:
            text = sample["text"]
            if not text.strip():
                continue
            ids = encoding.encode(text)
            buf.extend(ids)
            while len(buf) >= seq_len + 1:
                chunk = buf[: seq_len + 1]
                yield chunk
                buf = buf[seq_len:]
        # Drain remainder after dataset exhausted
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            yield chunk
            buf = buf[seq_len:]

    data_iter = iter(token_generator())

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    start_step = 1
    ckpt_path = ckpt_dir / "latest.pt"
    if ckpt_path.exists():
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        # Validate that the checkpoint config matches the current model architecture
        saved_cfg = ckpt["cfg"]
        for key in ("dim", "n_heads", "recurrent_layers", "n_experts", "vocab_size"):
            current = getattr(cfg, key)
            saved = getattr(saved_cfg, key)
            assert current == saved, (
                f"Config mismatch on resume: {key}={current} (script) vs {saved} (checkpoint). "
                f"Update the script to match or delete the checkpoint."
            )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        logger.info(f"Resumed at step {start_step}")
    else:
        logger.info("No checkpoint found — starting fresh")
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    total_tokens_seen = (start_step - 1) * tokens_per_step
    step_loss = 0.0
    t_start = time.perf_counter()

    for step in range(start_step, total_steps + 1):
        # --- accumulate gradients over micro-batches ---
        optimizer.zero_grad(set_to_none=True)

        for micro in range(grad_accum):
            # Build batch
            batch_ids = []
            for _ in range(micro_batch):
                try:
                    chunk = next(data_iter)
                except StopIteration:
                    logger.info("Dataset exhausted — stopping.")
                    _save_checkpoint(model, optimizer, step, cfg, ckpt_dir)
                    return
                batch_ids.append(chunk)

            x = torch.tensor([ids[:-1] for ids in batch_ids], device=device)
            y = torch.tensor([ids[1:] for ids in batch_ids], device=device)
            total_tokens_seen += x.numel()

            # Forward
            with torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else _nullcontext():
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            step_loss += loss.item()

        # --- optimizer step ---
        lr_step = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_step

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- logging ---
        if step % log_every == 0:
            elapsed = time.perf_counter() - t_start
            tok_per_sec = total_tokens_seen / elapsed if elapsed > 0 else 0
            logger.info(
                f"step {step:>6,}/{total_steps:,} | "
                f"loss={step_loss / (log_every * grad_accum):.4f} | "
                f"lr={lr_step:.2e} | "
                f"grad_norm={grad_norm:.2f} | "
                f"tok={total_tokens_seen / 1e9:.2f}B | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            step_loss = 0.0

        # --- checkpoint ---
        if step % ckpt_every == 0:
            _save_checkpoint(model, optimizer, step, cfg, ckpt_dir)

    # Final save
    _save_checkpoint(model, optimizer, total_steps, cfg, ckpt_dir)
    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Training complete. {total_tokens_seen / 1e9:.2f}B tokens in "
        f"{elapsed / 3600:.1f}h ({total_tokens_seen / elapsed:,.0f} tok/s avg)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


def _save_checkpoint(model, optimizer, step, cfg, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / ".tmp_latest.pt"
    final = ckpt_dir / "latest.pt"

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    torch.save(state, str(tmp))
    os.replace(str(tmp), str(final))
    CHECKPOINT_VOLUME.commit()


# ---------------------------------------------------------------------------
# Local entrypoint — kick off training
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    train.remote()
