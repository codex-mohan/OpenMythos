from open_mythos.main import MythosConfig

# Per-layer independent MoE — every recurrent layer has its own router and expert pool.
# Shared experts are same size as routed experts (Qwen-style, not n_per_tok× larger).
#
# Design ratios:
#   expert_dim = dim * 3//8  (37.5%, between Qwen 25% and DeepSeek-V3 28%)
#   n_shared_experts = 1
#   n_experts_per_tok = 4–8


def mythos_200m() -> MythosConfig:
    """~225M total / 125M active. dim=1024, 4 rec layers x 4 loops = 16 depth."""
    return MythosConfig(
        vocab_size=32000, dim=1024, n_heads=8, n_kv_heads=2, max_seq_len=4096,
        recurrent_layers=4, max_loop_iters=4, prelude_layers=3, coda_layers=3,
        attn_type="mla",
        kv_lora_rank=128, q_lora_rank=256,
        qk_rope_head_dim=32, qk_nope_head_dim=48, v_head_dim=48,
        n_experts=32, n_shared_experts=1, n_experts_per_tok=4, expert_dim=384,
        act_threshold=0.99, rope_theta=500000.0, lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """~3.4B total / 320M active (9.3%). dim=2048, 8 rec layers x 2 loops = 16 depth."""
    return MythosConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4, max_seq_len=4096,
        recurrent_layers=8, max_loop_iters=2, prelude_layers=2, coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256, q_lora_rank=512,
        qk_rope_head_dim=32, qk_nope_head_dim=64, v_head_dim=64,
        n_experts=128, n_shared_experts=1, n_experts_per_tok=4, expert_dim=512,
        act_threshold=0.99, rope_theta=500000.0, lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """~10.1B total / 1.10B active (10.9%). dim=3072, 10 rec layers x 3 loops = 30 depth."""
    return MythosConfig(
        vocab_size=32000, dim=3072, n_heads=24, n_kv_heads=6, max_seq_len=8192,
        recurrent_layers=10, max_loop_iters=3, prelude_layers=2, coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384, q_lora_rank=768,
        qk_rope_head_dim=32, qk_nope_head_dim=96, v_head_dim=96,
        n_experts=128, n_shared_experts=1, n_experts_per_tok=8, expert_dim=768,
        act_threshold=0.99, rope_theta=500000.0, lora_rank=8,
    )


def mythos_50b() -> MythosConfig:
    """~49.8B total / 4.81B active (9.7%). dim=4096, 20 rec layers x 4 loops = 80 depth."""
    return MythosConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8, max_seq_len=8192,
        recurrent_layers=20, max_loop_iters=4, prelude_layers=3, coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512, q_lora_rank=1024,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        n_experts=128, n_shared_experts=1, n_experts_per_tok=8, expert_dim=1536,
        act_threshold=0.99, rope_theta=500000.0, lora_rank=16,
    )


def mythos_100b() -> MythosConfig:
    """~101B total / 11.1B active (11.0%). dim=6144, 24 rec layers x 4 loops = 96 depth."""
    return MythosConfig(
        vocab_size=32000, dim=6144, n_heads=48, n_kv_heads=8, max_seq_len=1000000,
        recurrent_layers=24, max_loop_iters=4, prelude_layers=4, coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512, q_lora_rank=1536,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        n_experts=144, n_shared_experts=1, n_experts_per_tok=8, expert_dim=1536,
        act_threshold=0.99, rope_theta=1000000.0, lora_rank=32,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """~479B total / 21.7B active (4.5%). dim=12288, 32 rec layers x 5 loops = 160 depth."""
    return MythosConfig(
        vocab_size=100000, dim=12288, n_heads=96, n_kv_heads=16, max_seq_len=1000000,
        recurrent_layers=32, max_loop_iters=5, prelude_layers=4, coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024, q_lora_rank=3072,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        n_experts=128, n_shared_experts=1, n_experts_per_tok=8, expert_dim=3072,
        act_threshold=0.99, rope_theta=1000000.0, lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """~993B total / 44.5B active (4.5%). dim=16384, 40 rec layers x 6 loops = 240 depth."""
    return MythosConfig(
        vocab_size=100000, dim=16384, n_heads=128, n_kv_heads=16, max_seq_len=1000000,
        recurrent_layers=40, max_loop_iters=6, prelude_layers=6, coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024, q_lora_rank=4096,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        n_experts=128, n_shared_experts=1, n_experts_per_tok=8, expert_dim=4096,
        act_threshold=0.99, rope_theta=2000000.0, lora_rank=256,
        max_output_tokens=131072,
    )
