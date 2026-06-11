"""
Train OpenMythos on TinyStories locally (RTX 4050 Laptop, 6GB VRAM).

~117M variant (mythos_200m base: dim=1024, 4 rec layers x 4 loops = 16 depth,
24 experts top-4 + 1 shared, MLA, expert_dim=384, shared MoE counted once)
with the following 6GB-friendly tweaks:
  - vocab: gpt2 (50,257)
  - n_experts: 32 -> 24              (drop 8 routed experts)
  - max_seq_len: 4096 -> 512         (activation memory, default 512)
  - use_act: False                   (no ACT head)
  - grad_checkpoint: opt-in          (3x extra compute; only useful at >1024 ctx)

Default tokenizer is GPT-2 (50,257 vocab).  Override with --tokenizer.

Optimizer: --optimizer muon (default) uses split Muon (matrix params get
Newton-Schulz orthogonalized momentum at --muon-lr, 1D biases/norms get
AdamW at --lr).  --optimizer adamw uses pure AdamW for everything.

Usage:
    python training/train_tinystories.py                       # 1 epoch, 10K, Muon (default)
    python training/train_tinystories.py --optimizer adamw     # pure AdamW
    python training/train_tinystories.py --epochs 3            # 3 epochs
    python training/train_tinystories.py --seq-len 1024        # longer context
    python training/train_tinystories.py --muon-lr 0.01        # tune Muon LR
    python training/train_tinystories.py --grad-ckpt           # enable gradient checkpointing
    python training/train_tinystories.py --compile             # torch.compile (Win + 4050 = unstable)
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from contextlib import nullcontext

from open_mythos import OpenMythos, mythos_200m, MythosTokenizer


# ---------------------------------------------------------------------------
# TinyStories config — true 200M variant, 6GB-friendly overrides only
# ---------------------------------------------------------------------------

def tinystories_cfg(seq_len: int = 512, grad_checkpoint: bool = False) -> dict:
    """Returns a ~117M variant on 6GB VRAM (shared MoE counted once).

    Architecture (from mythos_200m base):
        dim=1024, attn=mla, rec=4x4=16, prel=3, coda=3,
        expert_dim=384, kv_lora=128, q_lora=256,
        24 routed experts (top-4) + 1 shared

    Memory-friendly overrides:
        vocab: gpt2 (50,257)
        n_experts: 32 -> 24
        max_seq_len: 4096 -> seq_len        (activations, default 512)
        use_act: False                      (no ACT head)
        grad_checkpoint: opt-in
        dropout: 0.0
    """
    cfg = mythos_200m().with_vocab("gpt2")
    cfg.n_experts = 24
    cfg.n_shared_experts = 1
    cfg.n_experts_per_tok = 4
    cfg.max_seq_len = seq_len
    cfg.dropout = 0.0
    cfg.use_act = False
    cfg.grad_checkpoint = grad_checkpoint
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OpenMythos on TinyStories")
    p.add_argument("--steps", type=int, default=0, help="Raw step count (0 = derive from --epochs)")
    p.add_argument("--epochs", type=int, default=1, help="Epochs over the loaded stories (default 1)")
    p.add_argument("--max-samples", type=int, default=10000, help="Stories to load (default 10K)")
    p.add_argument("--start-sample", type=int, default=0, help="Skip the first N stories of the dataset (use with --max-samples for fresh continuation)")
    p.add_argument("--fresh", action="store_true", help="Start training from scratch (ignore existing latest.pt)")
    p.add_argument("--init-from", type=str, default="", help="Path to a checkpoint .pt to initialize weights from (default: auto-resume from --ckpt-dir/latest.pt)")
    p.add_argument("--seq-len", type=int, default=512, help="Sequence length (default 512, 4x faster than 1024)")
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch size (default 2 — fits 6GB without grad_ckpt)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation (default 8, eff batch=16)")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (AdamW / 1D-group inside Muon)")
    p.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"],
                   help="Optimizer: 'muon' (default, 2D+ matrix params via Newton-Schulz + AdamW for 1D) or 'adamw' (everything via AdamW)")
    p.add_argument("--muon-lr", type=float, default=0.02, help="Peak LR for Muon matrix params (default 0.02)")
    p.add_argument("--muon-momentum", type=float, default=0.95, help="Muon momentum (default 0.95)")
    p.add_argument("--muon-wd", type=float, default=0.1, help="Muon weight decay (default 0.1)")
    p.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    p.add_argument("--ckpt-every", type=int, default=50, help="Checkpoint every N steps")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/tinystories", help="Checkpoint directory")
    p.add_argument("--eval-steps", type=int, default=0, help="Eval every N steps (0 = every epoch)")
    p.add_argument("--eval-batches", type=int, default=4, help="Eval batches (default 4, was 10)")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer model_id (default gpt2)")
    p.add_argument("--warmup", type=int, default=50, help="LR warmup steps")
    p.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing (3x compute cost)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (extra speedup, may be unstable on Win)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    # ------------------------------------------------------------------
    # Config + Tokenizer + Model
    # ------------------------------------------------------------------
    cfg = tinystories_cfg(seq_len=args.seq_len, grad_checkpoint=args.grad_ckpt)
    encoding = MythosTokenizer(args.tokenizer)
    assert encoding.vocab_size == cfg.vocab_size, (
        f"vocab mismatch: tokenizer={encoding.vocab_size} config={cfg.vocab_size}"
    )
    logger.info(f"Tokenizer: {encoding}")

    model = OpenMythos(cfg).to(device=device, dtype=amp_dtype if use_amp else torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {n_params:,} params ({n_trainable:,} trainable) | "
        f"rec={cfg.recurrent_layers}x{cfg.max_loop_iters}={cfg.recurrent_layers*cfg.max_loop_iters} | "
        f"experts={cfg.n_experts} (top-{cfg.n_experts_per_tok}+{cfg.n_shared_experts} shared) | "
        f"expert_dim={cfg.expert_dim} | "
        f"grad_ckpt={cfg.grad_checkpoint}"
    )

    if args.compile:
        if os.name == "nt":
            logger.warning(
                "torch.compile is not stable on Windows + RTX 4050 + Python 3.13 "
                "(CUDAGraphs overwrite errors in the recurrent block). Skipping."
            )
        else:
            logger.info("Compiling model with torch.compile (this can take 30-60s)...")
            model = torch.compile(model, mode="reduce-overhead")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    if args.optimizer == "muon":
        from open_mythos.muon import split_muon_adamw
        optimizer = split_muon_adamw(
            model,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_wd=args.muon_wd,
            adamw_lr=args.lr,
        )
        n_muon = sum(len(pg["params"]) for pg in optimizer.param_groups if pg.get("optimizer") == "muon")
        n_adamw = sum(len(pg["params"]) for pg in optimizer.param_groups if pg.get("optimizer") == "adamw")
        logger.info(
            f"Optimizer: Muon (split) | matrix params={n_muon} @ lr={args.muon_lr:.4f} "
            f"mom={args.muon_momentum} wd={args.muon_wd} | 1D params={n_adamw} @ lr={args.lr:.4f}"
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
            fused=True if device.type == "cuda" else False,
        )
        logger.info(f"Optimizer: AdamW | lr={args.lr:.4f} betas=(0.9, 0.95) wd=0.1")

    def lr_schedule(step: int, total: int, warmup: int, peak: float) -> float:
        """Linear warmup -> cosine decay to 10% of peak."""
        if step < warmup:
            return peak * step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))

    def set_lrs(step: int, total: int) -> tuple[float, float]:
        """Compute and apply LRs.  Returns (matrix_lr, adamw_lr) for logging."""
        if args.optimizer == "muon":
            muon_lr = lr_schedule(step, total, args.warmup, args.muon_lr)
            adamw_lr = lr_schedule(step, total, args.warmup, args.lr)
            for pg in optimizer.param_groups:
                if pg.get("optimizer") == "muon":
                    pg["lr"] = muon_lr
                else:
                    pg["lr"] = adamw_lr
            return muon_lr, adamw_lr
        adamw_lr = lr_schedule(step, total, args.warmup, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = adamw_lr
        return adamw_lr, adamw_lr  # (matrix_lr, adamw_lr) — same for AdamW

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if args.max_samples > 0:
        if args.start_sample > 0:
            logger.info(
                f"Loading TinyStories samples [{args.start_sample:,} .. "
                f"{args.start_sample + args.max_samples:,}] "
                f"({args.max_samples:,} stories)"
            )
        else:
            logger.info(f"Loading {args.max_samples:,} TinyStories samples...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        if args.start_sample > 0:
            ds = ds.skip(args.start_sample)
        texts = [s["text"] for s in ds.take(args.max_samples) if s["text"].strip()]
    else:
        logger.info("Loading full TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        texts = [s["text"] for s in ds if s["text"].strip()]

    # 80/20 train/val split
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    logger.info(f"Train: {len(train_texts):,} stories | Val: {len(val_texts):,} stories")

    # Pre-tokenize everything once then convert to numpy for fast slicing
    logger.info("Tokenizing training data...")
    train_buf = []
    for text in train_texts:
        train_buf.extend(encoding.encode(text))
    train_buf = np.array(train_buf, dtype=np.int64)
    logger.info(f"Train tokens: {len(train_buf):,}")

    logger.info("Tokenizing validation data...")
    val_buf = []
    for text in val_texts:
        val_buf.extend(encoding.encode(text))
    val_buf = np.array(val_buf, dtype=np.int64)
    logger.info(f"Validation tokens: {len(val_buf):,}")

    # ------------------------------------------------------------------
    # Resolve step count: --steps overrides --epochs
    # 1 epoch = process len(train_buf) tokens (one full pass)
    # ------------------------------------------------------------------
    tokens_per_step = args.batch_size * args.grad_accum * cfg.max_seq_len
    steps_per_epoch = max(1, math.ceil(len(train_buf) / tokens_per_step))
    if args.steps > 0:
        total_steps = args.steps
        effective_epochs = total_steps / steps_per_epoch
        logger.info(
            f"Using --steps override: {total_steps:,} steps "
            f"({effective_epochs:.2f} epochs)"
        )
    else:
        total_steps = max(1, steps_per_epoch * args.epochs)
        effective_epochs = float(args.epochs)
        logger.info(
            f"Using --epochs={args.epochs}: {steps_per_epoch:,} steps/epoch x "
            f"{args.epochs} = {total_steps:,} total steps"
        )

    def build_batch(buf, batch_size, seq_len):
        """Sample random fixed-length (input, target) chunks from a flat token buffer.

        CPU-side numpy vectorized indexing + pinned host tensor + non_blocking copy.
        """
        n = len(buf) - seq_len - 1
        if n <= 0:
            raise ValueError(f"buffer too small ({len(buf)}) for seq_len={seq_len}")
        starts = np.random.randint(0, n, size=batch_size)
        idx = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)
        arr = buf[idx]
        x = torch.from_numpy(arr[:, :-1].copy()).pin_memory().to(device, non_blocking=True)
        y = torch.from_numpy(arr[:, 1:].copy()).pin_memory().to(device, non_blocking=True)
        return x, y

    @torch.no_grad()
    def evaluate(n_batches: int = 4):
        model.eval()
        total_loss = torch.zeros((), device=device)
        total_tokens = 0
        for _ in range(n_batches):
            x, y = build_batch(val_buf, args.batch_size, cfg.max_seq_len)
            with torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext():
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            total_loss = total_loss + loss.detach() * y.numel()
            total_tokens += y.numel()
        model.train()
        return (total_loss / total_tokens).item()

    # ------------------------------------------------------------------
    # Checkpoint resume
    # ------------------------------------------------------------------
    start_step = 1
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.fresh:
        ckpt_path = os.path.join(args.ckpt_dir, "latest.pt")
        logger.info(f"--fresh set: ignoring existing checkpoint at {ckpt_path}")
    else:
        ckpt_path = args.init_from or os.path.join(args.ckpt_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_cfg = ckpt["cfg"]
            for key in ("dim", "recurrent_layers", "n_experts", "vocab_size", "max_seq_len"):
                assert getattr(cfg, key) == getattr(saved_cfg, key), (
                    f"Config mismatch on {key}: current={getattr(cfg, key)} ckpt={getattr(saved_cfg, key)}"
                )
            model.load_state_dict(ckpt["model"])
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state ({e}); starting with fresh optimizer")
            start_step = ckpt.get("step", 0) + 1
            logger.info(f"Resumed at step {start_step} (was trained for {ckpt.get('step', 0)} steps)")
        else:
            logger.info(f"No checkpoint at {ckpt_path} — starting fresh")

    logger.info(
        f"Training: {total_steps:,} steps | "
        f"batch={args.batch_size}×{args.grad_accum} | seq={cfg.max_seq_len} | "
        f"{tokens_per_step:,} tok/step | {effective_epochs:.2f} epochs"
    )

    # ------------------------------------------------------------------
    # Background log thread — keeps I/O off the training hot path
    # ------------------------------------------------------------------
    import queue as _queue
    import threading as _threading

    _logq: _queue.Queue = _queue.Queue()

    def _log_worker() -> None:
        while True:
            msg = _logq.get()
            if msg is None:
                break
            logger.info(msg)

    _log_thread = _threading.Thread(target=_log_worker, daemon=True)
    _log_thread.start()

    # ------------------------------------------------------------------
    # Graceful interrupt handler — saves checkpoint on Ctrl+C
    # ------------------------------------------------------------------
    _interrupted = False

    def _on_interrupt(signum, frame):
        nonlocal _interrupted
        _interrupted = True
        _logq.put("\n--- Ctrl+C received, saving checkpoint and exiting ---")

    _prev_handler = signal.signal(signal.SIGINT, _on_interrupt)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    total_tokens = (start_step - 1) * args.batch_size * args.grad_accum * cfg.max_seq_len
    step_loss = 0.0
    best_val_loss = float("inf")
    t_start = time.perf_counter()
    last_eval_step = start_step - 1

    _logq.put(
        f"Heartbeat: CUDA init + first forward can take 5-30s on a 4050. "
        f"Logging every {args.log_every} step(s)."
    )

    for step in range(start_step, total_steps + 1):
        if _interrupted:
            break

        optimizer.zero_grad(set_to_none=True)

        # Accumulate loss as a tensor — sync to Python only at log time
        step_loss = torch.zeros((), device=device)
        micro_count = 0
        for micro in range(args.grad_accum):
            x, y = build_batch(train_buf, args.batch_size, cfg.max_seq_len)
            total_tokens += x.numel()

            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(x)
                ce = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size), y.view(-1)
                )
                loss = ce / args.grad_accum

            loss.backward()
            step_loss = step_loss + ce.detach()
            micro_count += 1

        muon_lr, adamw_lr = set_lrs(step, total_steps)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch = (step - 1) // steps_per_epoch + 1
        epoch_frac = ((step - 1) % steps_per_epoch + 1) / steps_per_epoch

        if step % args.log_every == 0 or step == total_steps:
            loss_val = (step_loss / micro_count).item()
            elapsed = time.perf_counter() - t_start
            tok_sec = total_tokens / elapsed if elapsed > 0 else 0
            if args.optimizer == "muon":
                lr_str = f"lr_m={muon_lr:.2e}/a={adamw_lr:.2e}"
            else:
                lr_str = f"lr={adamw_lr:.2e}"
            _logq.put(
                f"step {step:>5,}/{total_steps:,} | "
                f"epoch {epoch}/{max(1, math.ceil(effective_epochs))} ({epoch_frac:>4.0%}) | "
                f"loss={loss_val:.4f} | "
                f"{lr_str} | grad={grad_norm:.2f} | "
                f"tok={total_tokens:,} | {tok_sec:,.0f} tok/s"
            )

        # Eval: every epoch by default, or every --eval-steps
        do_eval = (
            (args.eval_steps > 0 and step % args.eval_steps == 0)
            or (args.eval_steps == 0 and step % steps_per_epoch == 0 and step != last_eval_step)
        )
        if do_eval and val_buf:
            last_eval_step = step
            val_loss = evaluate(args.eval_batches)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss, "cfg": cfg},
                           os.path.join(args.ckpt_dir, "best.pt"))
            _logq.put(f"  eval loss={val_loss:.4f}  {'*best*' if improved else ''}")

        if step % args.ckpt_every == 0 or step == total_steps:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "cfg": cfg, "val_loss": best_val_loss},
                ckpt_path,
            )

        # Check for interrupt again here so we catch it during long steps
        if _interrupted:
            break

    # --- interrupt or normal completion ---
    if _interrupted:
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "step": step, "cfg": cfg, "val_loss": best_val_loss},
            ckpt_path,
        )
        _logq.put(f"Checkpoint saved at step {step}. Resumable from {ckpt_path}")

    signal.signal(signal.SIGINT, _prev_handler)

    elapsed = time.perf_counter() - t_start
    _logq.put(
        f"\n{'='*60}\nTRAINING {'INTERRUPTED' if _interrupted else 'COMPLETE'}\n"
        f"  steps      : {step:,}\n"
        f"  epochs     : {step / steps_per_epoch:.2f}\n"
        f"  tokens     : {total_tokens:,}\n"
        f"  wall time  : {elapsed / 60:.1f} min\n"
        f"  throughput : {total_tokens / elapsed:,.0f} tok/s\n"
        f"  best val   : {best_val_loss:.4f}\n"
        f"  checkpoint : {ckpt_path}\n"
        f"{'='*60}"
    )

    # Drain the log queue so final messages print before generation test
    _logq.put(None)
    _log_thread.join(timeout=5)

    if _interrupted:
        logger.info("Exiting after interrupt.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Quick generation test
    # ------------------------------------------------------------------
    logger.info("\n--- Generation test ---")
    model.eval()
    prompt_texts = [
        "Once upon a time, there was a little",
        "The cat and the dog were friends.",
    ]
    for prompt_text in prompt_texts:
        ids = torch.tensor([encoding.encode(prompt_text)], device=device)
        out_ids = model.generate(ids, max_new_tokens=50, temperature=0.8, top_k=40, n_loops=cfg.max_loop_iters)
        generated = encoding.decode(out_ids[0].tolist())
        logger.info(f"Prompt  : {prompt_text}")
        logger.info(f"Response: {generated}\n")


if __name__ == "__main__":
    main()
