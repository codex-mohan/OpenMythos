import torch
from open_mythos.main import OpenMythos, MythosConfig
from open_mythos import get_vocab_size, MythosTokenizer

attn_type = "mla"  # or "gqa"

# --- Option 1: fast vocab-size lookup (downloads tokenizer config only) ---
vocab_size = get_vocab_size("openai/gpt-oss-20b")
print(f"gpt-oss-20b vocab_size = {vocab_size:,}")

# --- Option 2: build config from a variant + tokenizer ---
cfg = MythosConfig(
    vocab_size=vocab_size,
    dim=256,
    n_heads=8,
    max_seq_len=128,
    recurrent_layers=2,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type=attn_type,
    n_kv_heads=2,
    kv_lora_rank=32,
    q_lora_rank=64,
    qk_rope_head_dim=16,
    qk_nope_head_dim=16,
    v_head_dim=16,
)

model = OpenMythos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"[{attn_type.upper()}] Parameters: {total:,}")

ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
print(f"[{attn_type.upper()}] Logits shape: {logits.shape}")

out = model.generate(ids, max_new_tokens=8, n_loops=8)
print(f"[{attn_type.upper()}] Generated shape: {out.shape}")

A = model.recurrent.injection.get_A()
rho = A.max().item()
print(f"[{attn_type.upper()}] Spectral radius rho(A) max: {rho:.4f} (must be < 1)")

# --- Verify tokenizer roundtrip ---
tok = MythosTokenizer()
text = "Hello, this is a test of the Mythos tokenizer."
ids = tok.encode(text)
decoded = tok.decode(ids)
assert decoded == text, f"Roundtrip failed: {text!r} != {decoded!r}"
print(f"Tokenizer roundtrip OK: {tok}")
