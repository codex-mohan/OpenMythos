"""
Load a trained OpenMythos checkpoint and generate stories.

Usage:
    python training/generate.py                              # interactive prompts
    python training/generate.py --prompt "Once upon a time"  # single prompt
    python training/generate.py --num 5                      # generate 5 random stories
"""

from __future__ import annotations

import argparse
import os

import torch

from open_mythos import OpenMythos, MythosTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stories from a trained checkpoint")
    p.add_argument("--ckpt", type=str, default="checkpoints/tinystories/best.pt",
                   help="Checkpoint path (falls back to latest.pt)")
    p.add_argument("--prompt", type=str, default="",
                   help="Prompt text (empty = interactive mode)")
    p.add_argument("--num", type=int, default=1, help="Number of stories to generate")
    p.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=40, help="Top-K sampling")
    p.add_argument("--n-loops", type=int, default=0, help="Loop depth (0 = use training value)")
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device):
    """Load model + tokenizer from checkpoint."""
    if not os.path.exists(ckpt_path):
        alt = ckpt_path.replace("best.pt", "latest.pt")
        if os.path.exists(alt):
            ckpt_path = alt
            print(f"Best checkpoint not found, using: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint at {ckpt_path} or {alt}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" not in ckpt:
        latest = os.path.join(os.path.dirname(ckpt_path), "latest.pt")
        if os.path.exists(latest):
            cfg = torch.load(latest, map_location="cpu", weights_only=False)["cfg"]
        else:
            raise KeyError("No 'cfg' in checkpoint and no latest.pt to recover from")
    else:
        cfg = ckpt["cfg"]
    step = ckpt["step"]
    val_loss = ckpt.get("val_loss", float("nan"))

    model = OpenMythos(cfg).to(device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = MythosTokenizer()
    return model, tokenizer, cfg, step, val_loss


def generate(model, tokenizer, prompt: str, cfg, max_tokens: int,
             temperature: float, top_k: int, n_loops: int, device: torch.device):
    """Generate a single completion."""
    ids = tokenizer.encode(prompt) if prompt else []
    if not ids:
        ids = [0]  # fallback to BOS token if empty prompt

    input_ids = torch.tensor([ids], device=device)
    n_loops = n_loops or cfg.max_loop_iters

    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        n_loops=n_loops,
    )
    return tokenizer.decode(out_ids[0].tolist())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    model, tokenizer, cfg, step, val_loss = load_model(args.ckpt, device)
    print(
        f"  Step: {step} | Val loss: {val_loss:.4f} | "
        f"Recurrent: {cfg.recurrent_layers}×{cfg.max_loop_iters} | "
        f"Vocab: {cfg.vocab_size:,}"
    )

    n_loops = args.n_loops or cfg.max_loop_iters
    print(f"  Temperature: {args.temperature} | Top-K: {args.top_k} | Loops: {n_loops}\n")

    # --- Non-interactive (--prompt given or --num > 1) ---
    if args.prompt or args.num > 1:
        for i in range(args.num):
            text = generate(model, tokenizer, args.prompt or "", cfg,
                            args.max_tokens, args.temperature, args.top_k,
                            n_loops, device)
            if args.num > 1:
                print(f"--- Story {i + 1} ---")
            print(text)
            print()
        return

    # --- Interactive mode ---
    print("Enter prompts (empty line to exit, Ctrl+C to quit).")
    print("Type '/random' for a random story with no prompt.\n")
    try:
        while True:
            prompt = input("> ").strip()
            if not prompt:
                break
            if prompt.lower() == "/random":
                prompt = ""

            text = generate(model, tokenizer, prompt, cfg,
                            args.max_tokens, args.temperature, args.top_k,
                            n_loops, device)
            print(f"\n{text}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
