"""Count trainable parameters for any MythosConfig variant (no model instantiation).

Usage:
    python param_count.py           # counts all variants
    python param_count.py 3b        # counts just mythos_3b
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from open_mythos.main import MythosConfig


def human_size(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n/1e12:.2f}T"
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    return f"{n/1e3:.1f}K"


def count_mla_params(cfg: MythosConfig) -> int:
    n = 0
    n += cfg.dim * cfg.q_lora_rank                           # q_down
    n += cfg.q_lora_rank * cfg.n_heads * cfg.qk_nope_head_dim  # q_up_nope
    n += cfg.q_lora_rank * cfg.n_heads * cfg.qk_rope_head_dim  # q_up_rope
    n += cfg.dim * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)   # kv_down
    n += cfg.kv_lora_rank * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)  # kv_up
    n += cfg.n_heads * cfg.v_head_dim * cfg.dim                 # wo
    return n


def count_gqa_params(cfg: MythosConfig) -> int:
    head_dim = cfg.dim // cfg.n_heads
    n = 0
    n += cfg.dim * cfg.n_heads * head_dim       # wq
    n += cfg.dim * cfg.n_kv_heads * head_dim     # wk
    n += cfg.dim * cfg.n_kv_heads * head_dim     # wv
    n += cfg.n_heads * head_dim * cfg.dim         # wo
    return n


def count_attn_params(cfg: MythosConfig) -> int:
    if cfg.attn_type == "mla":
        return count_mla_params(cfg)
    else:
        return count_gqa_params(cfg)


def count_dense_ffn_params(cfg: MythosConfig) -> int:
    inner = cfg.dim * 4 // 3
    return 3 * cfg.dim * inner  # gate, up, down


def count_single_expert_params(cfg: MythosConfig, expert_dim: int) -> int:
    return 3 * cfg.dim * expert_dim


def count_moe_params(cfg: MythosConfig) -> tuple[int, int, int, int]:
    """Returns (routed, shared, router, total)."""
    router = cfg.dim * cfg.n_experts
    routed = cfg.n_experts * count_single_expert_params(cfg, cfg.expert_dim)
    shared = cfg.n_shared_experts * count_single_expert_params(cfg, cfg.expert_dim)
    return routed, shared, router, routed + shared + router


def count_norm_params(dim: int) -> int:
    return dim  # RMSNorm has 1 learnable vector


def count_model_params(cfg: MythosConfig) -> dict:
    """Returns breakdown dict from config — no model instantiation needed."""

    # Embedding (with tied head, counted once)
    embed = cfg.vocab_size * cfg.dim

    # Prelude: dense blocks
    prelude_attn = cfg.prelude_layers * count_attn_params(cfg)
    prelude_ffn = cfg.prelude_layers * count_dense_ffn_params(cfg)
    prelude_norms = cfg.prelude_layers * count_norm_params(cfg.dim) * 2  # attn_norm + ffn_norm

    # Recurrent: per-layer attention + shared MoE + norms
    rec_attn = cfg.recurrent_layers * count_attn_params(cfg)
    routed, shared, router, moe_total = count_moe_params(cfg)
    rec_moe_routed = cfg.recurrent_layers * routed
    rec_moe_shared = cfg.recurrent_layers * shared
    rec_moe_router = cfg.recurrent_layers * router
    rec_norms = cfg.recurrent_layers * count_norm_params(cfg.dim) * 2  # attn_norm + ffn_norm
    rec_norm_final = count_norm_params(cfg.dim)  # self.norm in RecurrentBlock
    rec_lti = cfg.dim * 3  # log_A + B + log_dt
    rec_act = cfg.dim * 2  # w_pause + b_pause

    # Coda: dense blocks
    coda_attn = cfg.coda_layers * count_attn_params(cfg)
    coda_ffn = cfg.coda_layers * count_dense_ffn_params(cfg)
    coda_norms = cfg.coda_layers * count_norm_params(cfg.dim) * 2

    # Final norm (self.norm in OpenMythos)
    final_norm = count_norm_params(cfg.dim)

    # Totals
    total_attn = prelude_attn + rec_attn + coda_attn
    total_dense = prelude_ffn + coda_ffn
    total_moe_routed = rec_moe_routed
    total_moe_shared = rec_moe_shared
    total_moe_router = rec_moe_router
    total_norms = prelude_norms + rec_norms + rec_norm_final + coda_norms + final_norm
    total_other = rec_lti + rec_act

    total = (
        embed
        + total_attn
        + total_dense
        + total_moe_routed
        + total_moe_shared
        + total_moe_router
        + total_norms
        + total_other
    )

    return {
        "total": total,
        "embed": embed,
        "attention": total_attn,
        "moe_routed": total_moe_routed,
        "moe_shared": total_moe_shared,
        "moe_router": total_moe_router,
        "dense_ffn": total_dense,
        "norms": total_norms,
        "other": total_other,
    }


def count_active_params(cfg: MythosConfig) -> int:
    active = cfg.vocab_size * cfg.dim  # embed

    # Prelude + Coda: all params active
    prelude_coda_layers = cfg.prelude_layers + cfg.coda_layers
    for _ in range(prelude_coda_layers):
        active += count_attn_params(cfg)
        active += count_dense_ffn_params(cfg)

    # LTI + ACT
    active += cfg.dim * 3  # LTI
    active += cfg.dim * 2  # ACT

    # Recurrent: all attn active + only top-k MoE active
    for _ in range(cfg.recurrent_layers):
        active += count_attn_params(cfg)
        active += cfg.dim * cfg.n_experts  # router
        active += cfg.n_experts_per_tok * count_single_expert_params(cfg, cfg.expert_dim)  # routed
        active += cfg.n_shared_experts * count_single_expert_params(cfg, cfg.expert_dim)  # shared

    # Output head (weight-tied with embed)
    active += cfg.dim * cfg.vocab_size

    return active


def print_breakdown(name: str, cfg: MythosConfig, counts: dict, active: int):
    total = counts["total"]
    active_ratio = active / total * 100 if total > 0 else 0
    expert_instances = cfg.n_experts * cfg.recurrent_layers

    print(f"\n{'='*80}")
    print(f"  {name.upper()}  —  {human_size(total)} total / {human_size(active)} active  "
          f"({active_ratio:.1f}%)")
    print(f"{'='*80}")
    print(f"  {'Component':<30} {'Params':>12} {'%':>8}")
    print(f"  {'-'*50}")

    items = [
        ("Embedding", counts["embed"]),
        ("Attention (MLA/GQA)", counts["attention"]),
        ("MoE routed experts", counts["moe_routed"]),
        ("MoE shared experts", counts["moe_shared"]),
        ("MoE routers", counts["moe_router"]),
        ("Dense FFN (prelude/coda)", counts["dense_ffn"]),
        ("Layer norms", counts["norms"]),
        ("Other (LTI, ACT)", counts["other"]),
    ]
    for label, v in items:
        pct = v / total * 100 if total > 0 else 0
        print(f"  {label:<30} {human_size(v):>12} {pct:>7.1f}%")

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<30} {human_size(total):>12}")
    nea = active - cfg.vocab_size * cfg.dim  # non-embed active
    print(f"\n  Non-embed active: {human_size(nea)} ({nea/total*100:.1f}% of total)")
    print(f"\n  Config: dim={cfg.dim}, rec_layers={cfg.recurrent_layers}, "
          f"loops={cfg.max_loop_iters}")
    print(f"  Effective depth: {cfg.recurrent_layers} x {cfg.max_loop_iters} = "
          f"{cfg.recurrent_layers * cfg.max_loop_iters}")
    print(f"  MoE: {cfg.n_experts} experts/layer x {cfg.recurrent_layers} layers = "
          f"{human_size(expert_instances)} expert instances total")
    print(f"  Routing: top-{cfg.n_experts_per_tok} of {cfg.n_experts} "
          f"+ {cfg.n_shared_experts} shared")
    print(f"  expert_dim={cfg.expert_dim} ({cfg.expert_dim/cfg.dim*100:.0f}% "
          f"of hidden_dim={cfg.dim})")


def main():
    from open_mythos.variants import (
        mythos_200m, mythos_3b, mythos_10b, mythos_50b,
        mythos_100b, mythos_500b, mythos_1t,
    )

    VARIANTS = {
        "200m": mythos_200m, "3b": mythos_3b, "10b": mythos_10b,
        "50b": mythos_50b, "100b": mythos_100b, "500b": mythos_500b, "1t": mythos_1t,
    }

    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key not in VARIANTS:
            print(f"Unknown variant: {key}. Options: {sorted(VARIANTS)}")
            return
        names = [key]
    else:
        names = list(VARIANTS)

    print("\n" + "=" * 80)
    print("  PARAMETER COUNT — OpenMythos (shared MoE)")
    print("=" * 80)

    for name in names:
        cfg = VARIANTS[name]()
        counts = count_model_params(cfg)
        active = count_active_params(cfg)
        print_breakdown(name, cfg, counts, active)


if __name__ == "__main__":
    main()
