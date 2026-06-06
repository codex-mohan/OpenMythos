"""
Head-to-head generation: OpenMythos (looped) vs HGDN-Hybrid (gated delta).
Same prompts, similar generation params, printed side-by-side.
"""

from __future__ import annotations

import sys
import time
import warnings

import torch

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Load OpenMythos
# ============================================================
print("Loading OpenMythos...")
sys.path.insert(0, r"C:\Users\wwwmo\Development\DL\OpenMythos")
from open_mythos import OpenMythos, MythosTokenizer

m_ckpt = torch.load(
    r"C:\Users\wwwmo\Development\DL\OpenMythos\checkpoints\tinystories\latest.pt",
    map_location=device, weights_only=False,
)
m_cfg = m_ckpt["cfg"]
m_model = OpenMythos(m_cfg).to(device=device)
m_model.load_state_dict(m_ckpt["model"])
m_model.eval()
m_tok = MythosTokenizer()
m_params = sum(p.numel() for p in m_model.parameters())
print(f"  {m_params:,} params | val_loss={m_ckpt['val_loss']:.4f} | "
      f"{m_cfg.recurrent_layers}x{m_cfg.max_loop_iters}={m_cfg.recurrent_layers*m_cfg.max_loop_iters} depth | "
      f"vocab={m_cfg.vocab_size:,} | steps={m_ckpt['step']}")

@torch.no_grad()
def mythos_gen(prompt: str, max_tokens: int = 200, temp: float = 0.8) -> str:
    ids = m_tok.encode(prompt) if prompt else []
    if not ids:
        ids = [0]
    inp = torch.tensor([ids], device=device)
    out = m_model.generate(inp, max_new_tokens=max_tokens, temperature=temp, top_k=40,
                           n_loops=m_cfg.max_loop_iters)
    return m_tok.decode(out[0].tolist())

# ============================================================
# Load HGDN-Hybrid
# ============================================================
print("Loading HGDN-Hybrid...")
sys.path.insert(0, r"C:\Users\wwwmo\Development\DL\hgdn-llm")
from configs.hgdn_hybrid_config import HGDNHybridConfig
from models.hgdn_hybrid_model import HGDNHybridModel
from transformers import AutoTokenizer

h_ckpt = torch.load(
    r"C:\Users\wwwmo\Development\DL\hgdn-research-kit\checkpoints\hgdn_tinystories\best_model.pt",
    map_location=device, weights_only=False,
)
h_cfg = h_ckpt["config"]
h_model = HGDNHybridModel(h_cfg).to(device=device)
state = h_ckpt.get("model_state_dict", h_ckpt.get("final_model", h_ckpt))
h_model.load_state_dict(state, strict=False)
h_model.eval()
h_tok = AutoTokenizer.from_pretrained("gpt2")
h_params = sum(p.numel() for p in h_model.parameters())
print(f"  {h_params:,} params | val_loss={h_ckpt.get('val_loss','?'):.4f} | "
      f"{h_cfg.n_layers} layers | vocab={h_cfg.vocab_size:,} | epoch={h_ckpt.get('epoch','?')}")

@torch.no_grad()
def hgdn_gen(prompt: str, max_tokens: int = 200, temp: float = 0.8) -> str:
    if not prompt:
        prompt = h_tok.bos_token or ""
    inp = h_tok(prompt, return_tensors="pt", add_special_tokens=bool(prompt)).input_ids.to(device)

    generated = inp.clone()
    for _ in range(max_tokens):
        logits, _ = h_model(generated[:, -h_cfg.max_seq_len:])
        logits = logits[:, -1, :] / temp
        top_k = 40
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == h_tok.eos_token_id:
            break
    return h_tok.decode(generated[0], skip_special_tokens=True)

# ============================================================
# HEAD-TO-HEAD
# ============================================================
PROMPTS = [
    "Once upon a time, there was a little",
    "The cat and the dog were friends.",
    "The dragon was very",
]

for i, prompt in enumerate(PROMPTS):
    print(f"\n{'='*70}")
    print(f"PROMPT {i+1}: {prompt}")
    print(f"{'='*70}")

    t0 = time.time()
    m_text = mythos_gen(prompt)
    dt_m = time.time() - t0

    t0 = time.time()
    h_text = hgdn_gen(prompt)
    dt_h = time.time() - t0

    print(f"\n--- OpenMythos ({m_params/1e6:.0f}M, looped, {dt_m:.1f}s) ---")
    print(m_text.encode("ascii", errors="replace").decode())

    print(f"\n--- HGDN-Hybrid ({h_params/1e6:.0f}M, gated-delta, {dt_h:.1f}s) ---")
    print(h_text.encode("ascii", errors="replace").decode())
