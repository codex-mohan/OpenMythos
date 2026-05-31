# Training Plan: Looped Transformer with Latent Reasoning

## Architecture (Fixed)

The original single-layer recurrent block is replaced with a **multi-layer stack**:

```
Input → [Prelude: 2 dense blocks]
     → [Recurrent: 12 transformer blocks × 4–8 loops]  ← the fix
     → [Coda: 2 dense blocks]
     → [RMSNorm + LM Head]
```

| Component | Detail |
|---|---|
| Recurrent depth | 12 transformer blocks (MLA + MoE FFN), looped 4–8 times |
| Effective depth | 48–96 layers of computation |
| Unique parameters | Equivalent to ~14-block transformer (prelude + recurrent + coda) |
| Attention | MLA (DeepSeek-V2 style), compressed KV cache |
| FFN | Fine-grained MoE (64–128 experts, top-4 routing, 2 shared) |
| Loop differentiation | Sinusoidal loop-index embedding + LoRA depth adapter per iteration |
| Stability | LTI injection with guaranteed ρ(A) < 1 |
| Adaptive compute | ACT halting with ponder-cost regularization |

---

## Phase 1 — Language Pretraining (30B tokens)

**Goal**: Make the model speak coherent English. No reasoning expected. Short loops.

### Data

| Dataset | Tokens | Weight |
|---|---|---|
| FineWeb-Edu (sample-100BT) | 25B | 83% |
| OpenWebMath | 3B | 10% |
| StarCoder (the-stack-smol) | 2B | 7% |

Streaming, no epoch limit. Pack documents to sequence length 2048 with cross-document attention mask.

### Loop Schedule

```
Epoch proportion 0%–30%:  n_loops = 2
Epoch proportion 30%–60%: n_loops = 3
Epoch proportion 60%–100%: n_loops = 4
```

Short loops during pretraining. The model learns basic language with mostly feedforward computation. Loops are gradually introduced so the recurrent dynamics don't destabilize early training.

### Objective

Standard next-token prediction, cross-entropy loss on all positions.

```
L_pretrain = -1/N Σ log P(token_t | token_<t)
```

No intermediate supervision. No ACT ponder cost yet. The loop-index embedding and LoRA adapter are active but untrained for reasoning — they just learn to not break language modeling.

### Optimization

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8) |
| Weight decay | 0.1 |
| Peak LR | 3e-4 |
| Min LR | 3e-5 |
| Schedule | Linear warmup 2000 steps → cosine to min |
| Batch size | 1024 sequences × 2048 tokens = 2M tokens/step |
| Precision | bfloat16 |
| Gradient clipping | 1.0 (global norm) |
| Sequence length | 2048 |
| Total steps | 15,000 (30B tokens) |

### Checkpoint

Save every 1000 steps. Keep last 5. Run a small held-out eval (C4 validation) to monitor perplexity. This phase should produce a model with perplexity comparable to a standard 3B dense transformer at 30B tokens — baseline sanity check.

---

## Phase 2 — Supervised Reasoning Distillation (5B tokens)

**Goal**: Teach the model that loops correspond to reasoning progression.

This is the critical phase. Without it, the model treats loops as redundant computation. The model must learn that `hidden_state at loop t` maps to `progress through reasoning step t`.

### Step 2a — Generate Reasoning Traces

Use a strong teacher model (Claude Sonnet 4, GPT-5, DeepSeek-R1) to generate reasoning traces across diverse domains.

**Generation prompt template** (send to teacher):

```
Solve the following problem. Think step by step. After each reasoning step,
output the token <NEXT_STEP>. After the final step, output <FINAL>.

[problem]
```

**Domains and sources**:

| Domain | Source | Samples | Avg. Steps |
|---|---|---|---|
| Math word problems | GSM8K, MATH train split | 20K | 3–5 |
| Multi-hop QA | HotpotQA, 2WikiMultihopQA | 20K | 2–4 |
| Code reasoning | MBPP, HumanEval prompts (expanded) | 10K | 3–6 |
| Logical deduction | LogiQA, synthetic logic puzzles | 10K | 3–7 |
| Science reasoning | ARC-Challenge, SciQ | 10K | 2–4 |
| Long-context synthesis | NarrativeQA, QMSum | 10K | 3–5 |
| Theorem proving | LeanDojo (tactic prediction) | 10K | 5–10 |
| Planning | ALFWorld, synthetic planning | 10K | 4–8 |

Total: 100K samples. Each has: `[question, list of step texts, final answer]`.

### Step 2b — Extract Intermediate Representations

Run each reasoning trace through the teacher model. Extract the hidden state at the last token of each step delimiter `<NEXT_STEP>`. This gives a sequence of teacher hidden states:

```
H_teacher = [h_step1, h_step2, ..., h_stepK, h_final]
```

where K is the number of reasoning steps and each h is a vector in the teacher's hidden dimension.

Project these to the looped model's dimension via a learned linear map:

```
h_target(t) = W_proj · h_teacher(t)     where W_proj ∈ R^(dim_teacher × dim_student)
```

Train `W_proj` once via least squares on a held-out set, then freeze it.

### Step 2c — Train the Looped Model with Intermediate Supervision

For each sample, run the looped model for T = min(16, 2×K) loops. At each loop, extract the hidden state from the final recurrent layer.

**Loss function**:

```
L_total = L_final_token + α · L_state_alignment + β · L_halting
```

**L_final_token**: Cross-entropy on the final answer tokens (standard LM loss on the answer portion only).

**L_state_alignment** (the key term):

Each loop `t` is soft-aligned to a reasoning step via a dynamic time warping (DTW) loss:

```
For each loop t ∈ [1, T]:
  Find k = round(t * K / T)  — map loop index to nearest reasoning step
  L_state(t) = MSE(h_model(t), h_target(k))
```

The model is NOT forced to exactly match step k at loop t — that would be too rigid. Instead, a soft penalty encourages the trajectory of h_model to roughly follow the trajectory of h_teacher:

```
L_state_alignment = soft_dtw(h_model[1:T], h_target[1:K])
```

where soft-DTW is differentiable and allows the model to learn its own mapping between loop depth and reasoning progress. The only constraint is that the trajectory must be monotonic — loops progress forward through reasoning, not backward.

**L_halting** (ponder cost):

```
L_halting = λ · Σ cumulative_halting_prob / batch_size
```

where λ starts at 0.01 and ramps to 0.05 over training. This teaches the ACT mechanism that extra loops have a cost.

### Step 2d — Loop Schedule

```
Sub-epoch 0–40%:  n_loops = 4–8   (short reasoning chains)
Sub-epoch 40–70%: n_loops = 8–12  (medium chains)
Sub-epoch 70–100%: n_loops = 12–16 (long chains, depth curriculum)
```

### Step 2e — Optimization

| Parameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Peak LR | 1e-4 (lower than pretrain — we're refining) |
| Min LR | 1e-5 |
| Batch size | 256 sequences |
| Sequence length | 2048 (packed), but only answer tokens count toward L_final |
| α (state alignment weight) | 0.1 → 0.5 linear warmup over first 20% of steps |
| β (halting cost weight) | 0.01 → 0.05 linear warmup |
| λ (soft-DTW bandwidth) | 0.1 |
| Steps | ~2,500 (5B tokens) |

### Step 2f — What This Phase Achieves

After Phase 2, the model should show:
- Halting probability that correlates with reasoning difficulty (ACT is trained)
- Hidden state trajectory that progresses monotonically through reasoning space
- The ability to answer reasoning questions with varying loop depths

Sanity check: run the model on a held-out math problem at loops = 2, 4, 8, 16 and verify that accuracy improves with depth. If it doesn't, Phase 2 failed — the model didn't learn the loop→reasoning mapping.

---

## Phase 3 — Reinforcement Learning with Verifiable Rewards (10B tokens)

**Goal**: Optimize the model to use loops efficiently and correctly on problems where answers are automatically verifiable. No human labels needed. No intermediate supervision needed — the model now bootstraps from its Phase 2 initialization.

### Reinforcement Learning Setup

**Algorithm**: Group Relative Policy Optimization (GRPO). Simpler than PPO, no critic model needed, works well for LLM fine-tuning.

For each problem in a batch, the model generates N candidate answers (N=8). Each answer gets a verifiable reward. Advantages are computed within the group (relative to the group mean). The policy is updated to favor answers that got higher-than-average reward.

**Why GRPO and not PPO**: PPO requires a separate value model that estimates expected reward — this doubles memory and compute. GRPO uses the group mean as baseline, which is unbiased and requires zero additional parameters. It's what DeepSeek-R1 used and it works.

### Reward Design

The reward has three components with relative weights:

```
R = R_correctness + γ · R_depth_efficiency + δ · R_format
```

| Component | Weight | Value |
|---|---|---|
| R_correctness | 1.0 | +1.0 if answer verifiably correct, -0.5 if wrong |
| R_depth_efficiency | γ = 0.3 | +0.3 if halted in ≤25% of max loops, +0.15 if ≤50%, +0.0 if ≤75%, -0.1 if max loops |
| R_format | δ = 0.1 | +0.1 if answer is properly formatted, -0.5 if unparseable |

The depth efficiency bonus creates a gradient toward early halting. The model is rewarded for solving problems in fewer loops. The correctness reward dominates (weight 1.0) so the model never sacrifices accuracy for speed — it must be BOTH correct and efficient.

### Training Data (Verifiable Tasks Only)

| Task | Source | Reward Verification | Samples |
|---|---|---|---|
| Math | GSM8K, MATH, AMC, AIME | Exact match / sympy equivalence | 50K |
| Code execution | MBPP, HumanEval, APPS, LiveCodeBench | Unit test pass/fail | 30K |
| Formal verification | LeanDojo, miniF2F | Proof checker result | 10K |
| Symbolic reasoning | BIG-Bench (logical, temporal, spatial) | Pattern match | 10K |
| Tool use | ToolBench, BFCL | API call correctness check | 10K |
| Game/planning | ALFWorld, TextWorld | Environment reward | 10K |

Total: 120K samples. Each is a prompt without any CoT — just the problem statement. The model must figure out the reasoning internally via loops and output only the answer.

### Training Loop (GRPO)

```
For each batch of M problems:

  1. For each problem (i = 1..M):
     a. Sample N = 8 completions from the current policy π_θ
     b. Each completion: run the model with ACT enabled.
        Model runs 1..T loops, halts when cumulative_p > threshold.
        Then decodes the answer greedily.
     c. Compute reward R_i,n for each completion via the verifier.
     d. Compute advantage A_i,n = (R_i,n - mean(R_i,·)) / std(R_i,·)

  2. For each completion with A_i,n > 0 (better than group average):
     a. Replay the entire forward pass (same number of loops as original generation)
     b. Compute policy gradient:
        ∇L = A_i,n · Σ_{tokens in answer} ∇ log π_θ(token)
        Plus depth bonus: A_i,n · η · ∇ log π_θ(halted_at_depth)

  3. Apply KL penalty to prevent divergence from Phase 2 model:
     KL_div = 0.01 · KL(π_θ || π_phase2)
     Total loss = -policy_gradient + KL_div

  4. Update θ
```

### ACT-Specific RL Modifications

The halting mechanism needs special treatment in RL because the halting decision is discrete and non-differentiable.

**Option A — REINFORCE on halting depth**:

Treat the halting depth `d` as a sampled action. Compute the advantage of halting at depth `d` vs. the expected depth under the current policy:

```
A_depth_i,n = R_i,n - baseline_depth
baseline_depth = moving average of R_i,n weighted by halting depth
```

Add this term to the policy gradient. This directly optimizes the ACT halting probability toward depths that produce higher reward.

**Option B — Differentiable soft-halting during RL**:

During RL training only, replace the discrete halt decision with a soft weighted average over all loop depths:

```
h_output = Σ_{t=1..T} softmax(halting_logits)_t · h_t
```

This makes the entire depth axis differentiable. The REINFORCE approach (Option A) is simpler and more robust — use that.

### Hyperparameters

| Parameter | Value |
|---|---|
| Group size N | 8 |
| Batch size M | 32 (256 completions total) |
| Max loops T | 16 |
| Learning rate | 5e-6 (very low — policy gradients are high variance) |
| KL penalty coefficient | 0.01 |
| γ (depth efficiency weight) | 0.3 |
| δ (format weight) | 0.1 |
| η (depth advantage weight) | 0.05 |
| Total steps | ~5,000 (10B tokens worth of completions) |
| Max answer length | 512 tokens |
| Temperature (sampling) | 1.0 (exploration), anneal to 0.8 |
| GRPO epsilon (clipping) | 0.2 |

### Phase 3 Monitoring

Track per-step:
- Mean reward (should increase)
- Mean halting depth (should bifurcate: low for easy, high for hard)
- KL divergence from Phase 2 (should stay below 0.1)
- Accuracy per difficulty bucket (easy/medium/hard problems)

If KL divergence spikes above 0.2, reduce learning rate. If mean reward plateaus, increase temperature temporarily to encourage exploration. If halting depth collapses to always-max or always-min, adjust γ (the depth efficiency bonus).

---

## Phase 4 — Post-Training Alignment (Optional, 2B tokens)

Standard chat fine-tuning to make the model usable:

### Data
- UltraChat, ShareGPT, OpenHermes, Capybara — standard instruction-tuning datasets
- Mix in 20% of the Phase 2 reasoning data (without intermediate supervision, just question→answer pairs)
- Add system prompt: "You are a helpful assistant. Think carefully before answering."

### Training
- Standard SFT with cross-entropy on assistant tokens only
- n_loops fixed at 8 (not variable — chat responses are mostly straightforward)
- LR = 5e-6, 1 epoch

---

## Infrastructure Requirements

### Phase 1 (Pretraining, 30B tokens)
- 8× H100 (80GB) — approximately 10–14 days
- FSDP with HYBRID_SHARD for the MoE experts
- Expert parallelism across nodes if using 64+ GPUs

### Phase 2 (Reasoning Distillation, 5B tokens)
- 4× H100 (80GB) — approximately 2–3 days
- Requires access to a strong teacher model API (Anthropic/OpenAI/DeepSeek)
- ~$500–1000 in API costs for generating 100K reasoning traces

### Phase 3 (RL, 10B tokens of completions)
- 8× H100 (80GB) — approximately 5–7 days
- Most expensive phase: each GRPO step generates 256 completions at T=16 loops
- Verification costs are negligible (rule-based checks, unit tests)

### Total
- ~40K H100-hours
- ~$80K at $2/GPU-hour (spot pricing) or ~$120K on-demand
- ~3–4 weeks wall-clock with 8 GPUs

---

## Expected Outcomes

| Metric | Target |
|---|---|
| Pretraining PPL (C4) | ≤ 15.0 (comparable to 3B dense model) |
| GSM8K accuracy (Phase 2) | ≥ 40% at 8 loops (baseline: 3B model ~25%) |
| GSM8K accuracy (Phase 3) | ≥ 55% at adaptive depth (RL optimizes for hard problems) |
| Depth scaling | Accuracy(16 loops) > Accuracy(4 loops) by ≥ 10 points on math |
| Mean halting depth (easy) | 2–4 loops |
| Mean halting depth (hard) | 10–14 loops |
| Hallucination rate | ≤ 5% on verifiable tasks (RL penalizes wrong answers) |

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Phase 2 alignment loss doesn't converge | Increase α weight; reduce DTW bandwidth; try per-loop hard assignment instead of soft DTW |
| Phase 3 reward hacking (model learns to game verifier) | Add format reward; periodically audit completions manually |
| ACT collapses to constant depth | Increase γ (efficiency bonus); add explicit diversity bonus for depth variance |
| KL divergence explodes in RL | Lower LR to 1e-6; increase KL penalty; add early stopping if KL > 0.2 for 500 steps |
| Teacher model distillation is noisy | Use ensemble of 2–3 teachers; average their hidden states; measure cosine similarity between teachers to detect noise |
| Depth extrapolation doesn't emerge | Increase Phase 2 curriculum max loops; add contrastive depth pairs (force improvement from loop 4→8) |
| MoE routing collapses during RL | Freeze router weights after Phase 2; only update attention + FFN during Phase 3 |

---

## Ablation Checklist (Run These First on Small Scale)

Before committing to the full run, validate each phase on a 100M-parameter model with 4 recurrent layers:

1. **Phase 1 only**: Verify looped model reaches comparable PPL to dense baseline at same params + token budget
2. **Phase 1+2 (no RL)**: Verify depth scaling on math — accuracy improves with more loops
3. **Phase 1+2 (with ACT)**: Verify halting distribution diverges — easy problems halt earlier
4. **Phase 1+2+3 (full)**: Verify RL improves over SFT alone on verifiable tasks
5. **Contrastive depth ablation**: Compare with vs. without contrastive depth pairs in Phase 2
6. **Multi-layer ablation**: Compare 12 recurrent layers vs. 1 recurrent layer (the original architecture). This validates the core architectural fix.

Each ablation should take < 24 hours on a single 8-GPU node. Don't skip this — debugging at scale is prohibitively expensive.
