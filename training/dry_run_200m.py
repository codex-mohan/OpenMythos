"""
Local dry-run script to verify the training loop before deploying to Modal.

Exercises the full pipeline — tokenizer, dataset streaming, model forward+backward,
optimizer step, LR schedule, checkpoint save/load — with a tiny config that fits
on an RTX 4050 (6 GB VRAM).

Usage:
    python training/dry_run_200m.py                 # full dry run (~2 min)
    python training/dry_run_200m.py --steps 10      # even shorter
    python training/dry_run_200m.py --device cpu    # fallback if no GPU
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from contextlib import nullcontext

from open_mythos import OpenMythos, mythos_200m, MythosTokenizer
from open_mythos.muon import split_muon_adamw


# ---------------------------------------------------------------------------
# Tiny dry-run config — shrinks model & data to fit laptop GPU
# ---------------------------------------------------------------------------

DRY_RUN_OVERRIDES = dict(
    recurrent_layers=2,   # was 8
    max_loop_iters=2,      # was 4 → effective depth = 4
    prelude_layers=1,      # was 4
    coda_layers=1,         # was 4
    n_experts=8,           # was 30
    n_experts_per_tok=2,   # was 4
    max_seq_len=256,
    use_act=False,
    dropout=0.0,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local dry-run of the Modal training loop")
    p.add_argument("--steps", type=int, default=50, help="Number of training steps")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    p.add_argument("--ckpt-dir", type=str, default="/tmp/mythos_dry_run", help="Checkpoint directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    # --- Config ---
    cfg = mythos_200m().with_vocab("openai/gpt-oss-20b")
    for k, v in DRY_RUN_OVERRIDES.items():
        setattr(cfg, k, v)
    cfg.max_seq_len = args.seq_len

    # --- Tokenizer ---
    encoding = MythosTokenizer()
    assert encoding.vocab_size == cfg.vocab_size, (
        f"Mismatch: tokenizer={encoding.vocab_size} config={cfg.vocab_size}"
    )
    logger.info(f"Tokenizer: {encoding}")

    # --- Model ---
    logger.info("Building model...")
    model = OpenMythos(cfg).to(device=device, dtype=amp_dtype if use_amp else torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_depth = cfg.recurrent_layers * cfg.max_loop_iters
    logger.info(
        f"Params: {n_params:,} total ({n_trainable:,} trainable) | "
        f"effective depth = {cfg.recurrent_layers}×{cfg.max_loop_iters} = {eff_depth}"
    )

    # --- Optimizer (Muon for 2D+ weights, AdamW for 1D biases/norms) ---
    optimizer = split_muon_adamw(model, muon_lr=0.02, muon_momentum=0.95, muon_wd=0.1)

    def get_lr(step: int, total: int, warmup: int = 10) -> float:
        muon_peak = 0.02
        muon_min = 0.002
        adamw_peak = 0.004
        adamw_min = 0.0004
        if step < warmup:
            frac = step / max(1, warmup)
        else:
            progress = (step - warmup) / max(1, total - warmup)
            frac = 0.5 * (1.0 + math.cos(math.pi * progress))
        return muon_peak * frac + muon_min * (1 - frac), adamw_peak * frac + adamw_min * (1 - frac)

    def set_lr(step: int, total: int, warmup: int = 10):
        muon_lr, adamw_lr = get_lr(step, total, warmup)
        for pg in optimizer.param_groups:
            if pg.get("optimizer") == "muon":
                pg["lr"] = muon_lr
            else:
                pg["lr"] = adamw_lr
        return muon_lr, adamw_lr

    # --- Dataset — streaming FineWeb-Edu ---
    logger.info("Loading dataset (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    def token_generator():
        buf: list[int] = []
        for sample in ds:
            text = sample["text"]
            if not text.strip():
                continue
            ids = encoding.encode(text)
            buf.extend(ids)
            while len(buf) >= args.seq_len + 1:
                chunk = buf[: args.seq_len + 1]
                yield chunk
                buf = buf[args.seq_len:]
        while len(buf) >= args.seq_len + 1:
            chunk = buf[: args.seq_len + 1]
            yield chunk
            buf = buf[args.seq_len:]

    data_iter = iter(token_generator())

    # --- Checkpoint resume test ---
    start_step = 1
    ckpt_path = os.path.join(args.ckpt_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1

    # --- Training loop ---
    model.train()
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    total_steps = start_step + args.steps - 1
    total_tokens = 0
    step_loss = 0.0
    t_start = time.perf_counter()

    logger.info(
        f"Starting dry run: {args.steps} steps | "
        f"batch={args.batch_size}×{args.grad_accum} acc | "
        f"seq_len={args.seq_len} | {tokens_per_step:,} tok/step"
    )

    for step in range(start_step, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            batch_ids = []
            for _ in range(args.batch_size):
                try:
                    chunk = next(data_iter)
                except StopIteration:
                    logger.warning("Dataset exhausted — stopping early.")
                    _save(model, optimizer, step, ckpt_path)
                    _report(t_start, total_tokens)
                    return
                batch_ids.append(chunk)

            x = torch.tensor([ids[:-1] for ids in batch_ids], device=device)
            y = torch.tensor([ids[1:] for ids in batch_ids], device=device)
            total_tokens += x.numel()

            amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, cfg.vocab_size), y.view(-1)
                ) / args.grad_accum

            loss.backward()
            step_loss += loss.item()

        muon_lr, adamw_lr = set_lr(step, total_steps)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == total_steps:
            elapsed = time.perf_counter() - t_start
            tok_sec = total_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                f"step {step:>4}/{total_steps} | "
                f"loss={step_loss / (10 * args.grad_accum):.4f} | "
                f"lr={muon_lr:.3f}/{adamw_lr:.4f} | grad_norm={grad_norm:.2f} | "
                f"tok={total_tokens:,} | {tok_sec:,.0f} tok/s"
            )
            step_loss = 0.0

        if step % 20 == 0:
            _save(model, optimizer, step, ckpt_path)

    _save(model, optimizer, total_steps, ckpt_path)

    # --- Verify checkpoint roundtrip ---
    logger.info("Verifying checkpoint roundtrip...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2 = OpenMythos(cfg).to(device=device, dtype=amp_dtype if use_amp else torch.float32)
    model2.load_state_dict(ckpt["model"])

    with torch.no_grad():
        x_test = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
        out1 = model(x_test)
        out2 = model2(x_test)
        max_diff = (out1.float() - out2.float()).abs().max().item()
        assert max_diff < 1e-3, f"Checkpoint roundtrip mismatch! max_diff={max_diff:.6f}"
    logger.info(f"Checkpoint roundtrip OK (max_diff={max_diff:.6f})")

    _report(t_start, total_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(model, optimizer, step, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )


def _report(t_start: float, total_tokens: int):
    elapsed = time.perf_counter() - t_start
    logger.info(
        f"\n{'='*50}\n"
        f"DRY RUN COMPLETE\n"
        f"  tokens processed : {total_tokens:,}\n"
        f"  wall time        : {elapsed:.1f}s\n"
        f"  avg throughput   : {total_tokens / elapsed:,.0f} tok/s\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    main()
