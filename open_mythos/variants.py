from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA attention × recurrent_layers + shared MoE
#   Effective depth = recurrent_layers × max_loop_iters
#   MoE is shared across all recurrent layers — each layer contributes its own attention weights (MLA)
#   and routes to different experts within the same shared pool due to varying hidden states.


def mythos_200m() -> MythosConfig:
    """200M parameter config. dim=1024, 8 rec layers × 4 loops = 32 effective depth, 30 experts, 4k context.

    Budget breakdown (~203M total):
        embed + tied head : ~33M
        prelude (4 blocks) : ~21M   (dense MLA + SwiGLU FFN)
        coda   (4 blocks)  : ~21M   (dense MLA + SwiGLU FFN)
        recurrent (8 layers): ~128M (shared MoE: 30 routed + 2 shared experts, 8×MLA attention)
    Comparable to GPT-2 Small (124M dense) but with 32 effective depth and MoE breadth.
    """
    return MythosConfig(
        vocab_size=32000,
        dim=1024,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=4096,
        recurrent_layers=8,
        max_loop_iters=4,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=48,
        v_head_dim=48,
        n_experts=30,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1024,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_1b() -> MythosConfig:
    """1B parameter config. dim=2048, 8 rec layers × 2 loops = 16 effective depth, 64 experts, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        recurrent_layers=8,
        max_loop_iters=2,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. dim=3072, 12 rec layers × 3 loops = 36 effective depth, 64 experts, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        recurrent_layers=12,
        max_loop_iters=3,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. dim=4096, 16 rec layers × 4 loops = 64 effective depth, 128 experts, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        recurrent_layers=16,
        max_loop_iters=4,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythos_50b() -> MythosConfig:
    """50B parameter config. dim=6144, 20 rec layers × 4 loops = 80 effective depth, 256 experts, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        recurrent_layers=20,
        max_loop_iters=4,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythos_100b() -> MythosConfig:
    """100B parameter config. dim=8192, 24 rec layers × 4 loops = 96 effective depth, 256 experts, 1M context."""
    return MythosConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        recurrent_layers=24,
        max_loop_iters=4,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. dim=12288, 32 rec layers × 5 loops = 160 effective depth, 512 experts, 1M context."""
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        recurrent_layers=32,
        max_loop_iters=5,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. dim=16384, 40 rec layers × 6 loops = 240 effective depth, 512 experts, 1M context."""
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        recurrent_layers=40,
        max_loop_iters=6,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
