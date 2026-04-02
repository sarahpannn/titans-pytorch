# Bug Squasher Log — TitanLLaMA Hybrid Architecture

## Bug 1: Double RMSNorm Before Attention (FIXED)

**Symptom**: LM loss ~12.9 (worse than random), distillation loss won't converge.

**Root cause**: The student applies **two** RMSNorms before the QKV projection, while
the teacher (HF LlamaAttention) applies only one.

**Where it happens**:

1. `titan_llama.py` — `TitanLLaMADecoderLayer.forward()` (line ~446):
   ```python
   hidden_states = self.input_layernorm(hidden_states)   # norm #1 — teacher's LEARNED weights
   ```

2. `titans_pytorch/mac_transformer.py` — `SegmentedAttention.forward_flex()` (line ~311):
   ```python
   seq = self.norm(seq)   # norm #2 — default ALL-ONES weights (nn.RMSNorm)
   ```

The teacher path (HF `LlamaDecoderLayer`) only does `input_layernorm` → `self_attn`,
with no internal norm inside `LlamaAttention`.

**Why it matters**: The second norm (all-ones weights) re-normalizes already-normalized
output, destroying the learned per-channel scaling from `input_layernorm`. Every layer
diverges from the teacher, making distillation impossible and the LM head (trained for
the teacher's distribution) produce garbage logits.

**Fix applied** (Option A):

1. Restored `input_layernorm` in `TitanLLaMADecoderLayer.forward()` (titan_llama.py:446).
2. Added `pre_normed` flag to `SegmentedAttention.__init__()` (mac_transformer.py:191).
3. All three forward paths (`forward_inference`, `forward_flex`, `forward`) skip
   `self.norm` when `pre_normed=True`.
4. `TitanLLaMAAttention` passes `pre_normed=True` when constructing `SegmentedAttention`
   (titan_llama.py:286).

---

## Bug 2: Value Residual Mixing — Random Init + Frozen (FIXED)

**Symptom**: LM loss catastrophically high even with correct norm.

**Root cause**: `SegmentedAttention.to_learned_v_mix` is randomly initialized and was
frozen during training. This module (which doesn't exist in the teacher) lerps the
current layer's V with the previous layer's V:

```python
# mac_transformer.py:207-211
self.to_learned_v_mix = nn.Sequential(
    nn.Linear(dim, heads),         # random init
    Rearrange('b n h -> b h n 1'),
    nn.Sigmoid()                   # outputs ~0.5 with random input
) if accept_value_residual else None

# mac_transformer.py:320-322  (in forward_flex)
mix = self.to_learned_v_mix(seq)       # ~0.5
v = v.lerp(value_residual, mix)        # 50% current V, 50% previous layer's V
```

Every layer after layer 0 has `accept_value_residual=True`, so attention values are
~50% corrupted with the wrong layer's values. The teacher (HF LlamaAttention) has no
such mixing.

**In `freeze_backbone`**: The parameter name (`segmented_attn.to_learned_v_mix.0.weight`)
didn't match any trainable pattern (`persistent_memory`, `.lora.`, `neural_memory`),
so it was frozen.

**Fix applied** (two parts):

1. **Zero-init in `_load_llama_weights`** (titan_llama.py ~line 1029):
   ```python
   v_mix = titan_layer.self_attn.segmented_attn.to_learned_v_mix
   if v_mix is not None:
       nn.init.zeros_(v_mix[0].weight)
       nn.init.constant_(v_mix[0].bias, -10.0)
   ```
   `sigmoid(-10) ≈ 0.00005`, so `v.lerp(vr, ~0) ≈ v` — starts equivalent to teacher.

2. **Made trainable in `freeze_backbone`** (titan_llama.py ~line 1058):
   ```python
   if "to_learned_v_mix" in name:
       param.requires_grad = True
       continue
   ```

---

## Bug 3: MemoryMLP Averages Away Batch Dimension During Retrieval (FIXED)

**Symptom**: Loss doesn't decrease meaningfully. Neural memory fails to learn
sequence-specific patterns.

**Root cause**: During `retrieve_memories`, `functional_call` substitutes the
MemoryMLP's weight parameters with TTT-updated weights of shape
`(batch, dim_in, dim_out)` — one set per sequence in the micro-batch. But
`MemoryMLP.forward()` sees `ndim == 3` and calls `.mean(dim=0)`, collapsing all
batch items into a single averaged weight matrix.

**Where it happens**:

1. `titans_pytorch/memory_models.py` — `MemoryMLP.forward()` (line ~121):
   ```python
   w = weight
   if w.ndim == 3:
       w = w.mean(dim=0)   # ← AVERAGES OVER BATCH
   x = x @ w
   ```

2. Same file — `LayerNorm.forward()` (line ~35):
   ```python
   while gamma.ndim > 1: gamma = gamma.mean(dim=0)   # ← also batch-averaged
   ```

**Why it matters**: Every sequence in the micro-batch retrieves from the **same
averaged memory** instead of its own sequence-specific memory. The per-sequence
specialization that TTT provides is completely destroyed. The memory can only
learn patterns useful *on average* across all batch items — for diverse text from
SlimPajama, this largely washes out any useful signal.

**How it got here**: The original `_apply_memory_linear` (commented out at line 124)
had per-head batched matmul via `einsum("bhd,hdf->bhf", ...)`, but was replaced
with the simpler `x @ w` path. The `mean(dim=0)` was added to avoid shape errors,
but it silently produces wrong results. This bug is a direct consequence of Bug 4
(see below) — the `accum_updates` change broke dimensional alignment, and
`mean(dim=0)` was patched in to suppress the resulting shape error.

**Fix applied**:

1. `MemoryMLP.forward()` (memory_models.py:~95): Removed `reshape` flattening and
   `w.mean(dim=0)`. Now uses `torch.bmm(x, weight)` when both x and weight are 3D
   (batched matmul), and `x @ weight` for the standard 2D path.
2. `LayerNorm.forward()` (memory_models.py:~25): Replaced
   `while gamma.ndim > 1: gamma = gamma.mean(dim=0)` with
   `gamma.unsqueeze(1)` when gamma is 2D from `functional_call`, preserving
   per-batch-item scaling.

---

## Bug 4: `accum_updates` Discards All But Last Weight Snapshot — Breaks Causality and Retrieval Alignment (FIXED)

**Symptom**: Non-causal information leakage during training; dimensional mismatch
between weight snapshots and query chunks in `retrieve_memories`.

**Root cause**: `accum_updates` was modified to only keep the final TTT weight
snapshot instead of accumulating all per-chunk snapshots.

**Where it happens**:

`titans_pytorch/neural_memory.py` — `NeuralMemory.forward()` (line ~997):
```python
def accum_updates(past_updates, future_updates):
    if not exists(future_updates):
        return past_updates
    return TensorDict({
        name: upd[:, -1:].contiguous()      # ← only keeps LAST timestep
        for name, upd in future_updates.items()
    })
```

The original code (commented out below it) accumulated all timesteps:
```python
# cat((past_update[:, :-1], future_update), dim=1)
```

**Why it matters — two consequences**:

1. **Non-causal information leakage**: With `neural_memory_batch_size=64` and
   `chunk_size=64`, the store loop runs 16 iterations (1024 tokens / 64 per batch).
   Each produces a `(bh, 1, ...)` update. After all 16 iterations, only the
   **final** weight snapshot survives. Tokens at position 0 retrieve using weights
   trained on positions 0–1023 — future information leaks backward. For causal LM,
   the model learns to "cheat" during training but can't at inference.

2. **Dimension mismatch triggers Bug 3**: After rearranging, weights have shape
   `(batch*heads, ...)` = `(8, ...)` but queries have
   `(batch*heads*n_chunks, chunk_size, dim)` = `(136, 64, 2048)`. This 8-vs-136
   mismatch is what triggers `MemoryMLP.mean(dim=0)`. With the original full-timestep
   accumulation, weights would be `(batch*heads*n_chunks, ...)` — matching queries.

**Bugs 3 and 4 share a root cause**: The `accum_updates` change broke dimensional
alignment in `retrieve_memories`. The `mean(dim=0)` hack in MemoryMLP was then
added to suppress the shape error, destroying per-batch-item memory in the process.

**Fix applied**:

1. Restored the original full-timestep accumulation in `accum_updates`
   (neural_memory.py:~999): `cat((past_update[:, :-1], future_update), dim=1)`
   instead of `upd[:, -1:].contiguous()`. Updates now have shape
   `(bh, n_chunks, ...)` matching query chunks in `retrieve_memories`.
2. Restored the `if not exists(past_updates): return future_updates` early return
   that had been commented out.

---

## Bug 5: `warmup_steps` CLI Argument Silently Ignored (OPEN — MODERATE)

**Symptom**: Learning rate schedule doesn't match user expectations.

**Root cause**: `create_model_and_optimizer` hardcodes warmup as 10% of total steps,
ignoring `config.warmup_steps`.

**Where it happens**:

`train_titan_llama.py` — `create_model_and_optimizer()` (line ~413):
```python
warmup_steps = int(config.total_steps * 0.1)   # ← ignores config.warmup_steps
```

The user passes `--warmup_steps 2000`, stored in `config.warmup_steps`, but this
line overrides it. With `total_tokens=10M`, `batch_size=64`, `seq_length=1024`:

- `total_steps = 10M / (64 × 1024) = 152`
- Actual warmup = `int(152 * 0.1)` = **15 steps** (not 2000)

Given only 152 total steps, 2000 wouldn't work anyway — but the config field is
misleading dead code.

---

## Bug 6: `distillation_layers` Includes Out-of-Range Layer 16 (OPEN — MINOR)

**Symptom**: One distillation layer silently dropped.

**Root cause**: With `num_layers=16`, valid layer indices are 0–15. The training
script specifies layer 16.

**Where it happens**:

`train_lora_slimpajama.sh` (line ~22):
```bash
--distillation_layers "3,4,7,8,11,12,15,16"
```

In `compute_attention_distillation_loss`, the `if s_idx >= len(student_hiddens)`
guard skips it silently. Should be `15` instead of `16`, or the last entry removed.

---

## Bug 7: `backbone_model` Registered as Submodule (OPEN — MINOR)

**Symptom**: Teacher model put into train mode; frozen params included in
gradient clipping iteration.

**Root cause**: PyTorch's `Module.__setattr__` auto-registers `Module` assignments.

**Where it happens**:

`titan_llama.py` — `from_pretrained_llama()` (line ~975):
```python
model.backbone_model = base_model
```

This makes `backbone_model` a submodule, so `model.train()` (called at
`train_titan_llama.py:772`) also puts the teacher into training mode, and
`model.parameters()` includes all ~1B teacher parameters. For Llama specifically
(no dropout/batchnorm) this doesn't break correctness, but it wastes time during
gradient clipping and is a latent risk.

**Fix**: Store the teacher outside the Module hierarchy, e.g., assign to a
non-Module attribute or use `object.__setattr__`.

---

## Bug 8: `torch.set_default_device("cuda")` Never Reset (OPEN — MINOR)

**Symptom**: Global side effect — all subsequent tensor allocations default to CUDA.

**Where it happens**:

`titan_llama.py` — `from_pretrained_llama()` (line ~954):
```python
torch.set_default_device("cuda")
```

Called but never reverted. Doesn't cause errors in this training script but is a
landmine for downstream code.

---

## Bug 9: Indentation Bug in `create_model_and_optimizer` (OPEN — COSMETIC)

**Symptom**: O(n²) parameter iteration; bucket counts printed once per parameter.

**Where it happens**:

`train_titan_llama.py` (lines ~358–374): The bucket-count print loop, the
`neural_memory_params = []` reset, and the inner
`for name, param in model.named_parameters()` loop are all indented inside the
outer `for name, p in model.named_parameters()` loop.

Functionally correct because the inner loop rebuilds the lists from scratch each
time, but extremely wasteful.

---

## Bug 10: RoPE Theta Mismatch — SegmentedAttention Uses Default 10000 (FIXED)

**Symptom**: Loss ~9–11 even after Bugs 1–4 fixed.

**Root cause**: `SegmentedAttention` creates `RotaryEmbedding(dim_head)` with
default `theta=10000`, but LLaMA 3.1/3.2 uses `rope_theta=500000`. The config
value was stored but never passed through.

**Where it happens**:

1. `titans_pytorch/mac_transformer.py` — `SegmentedAttention.__init__()`:
   `RotaryEmbedding(dim_head)` used default theta.

2. `titan_llama.py` — `TitanLLaMAAttention.__init__()`:
   Did not pass `rope_theta` to `SegmentedAttention`.

**Fix applied**:

1. Added `rope_theta` parameter to `SegmentedAttention.__init__()`, passed to
   `RotaryEmbedding(dim_head, theta=rope_theta)`.
2. `TitanLLaMAAttention` passes `rope_theta=config.rope_theta` to the constructor.

---

## Bug 11: Neural Memory Outputs Random Noise at Init — No Zero-Gate (FIXED)

**Symptom**: Loss ~9–11. Pretrained backbone performance destroyed from step 0.

**Root cause**: The neural memory module has no mechanism to start with zero
contribution. At initialization:

- `to_queries` = `LinearNoBias(dim, dim)` — random init
- `MemoryMLP` weights — random init (then TTT-updated with random keys/values)
- `retrieve_gate = None` because `heads=1`
- `combine_heads = Identity()` because `heads=1`

The neural memory output `≈ random_proj(hidden_states) + LayerNorm(random_MLP(...))`
is on the same scale as hidden_states and added directly to the residual at layers
3, 7, 11 — completely corrupting pretrained features.

**Why it matters**: Unlike the value residual mixing (Bug 2, fixed with
`sigmoid(-10) ≈ 0`), there was no zero-init mechanism. With a pretrained backbone,
the model starts at random-quality loss instead of near-teacher loss.

**Fix applied** (KV-prepend approach):

Replaced additive injection (`residual += retrieved_memory`) with **attention
KV-prepend**. Retrieved memory is pooled to `num_neural_mem_kv_tokens` tokens
(adaptive avg pool), projected to K,V in head space via a **zero-initialized**
`mem_to_kv` linear, and prepended to attention K,V alongside persistent memory
tokens. The attention mechanism naturally decides how much to use memory.

Benefits over additive injection:
1. Zero-init is natural — zero K means zero dot products, zero V means zero
   contribution. No noise at init, pretrained behavior preserved.
2. Attention softmax acts as a natural gate — no need for explicit `neural_memory_scale`.
3. Mirrors the existing `persistent_memory` mechanism in `SegmentedAttention`.

Files modified:
- `titan_llama.py`: `TitanLLaMADecoderLayer` pools + projects retrieved memory to
  KV, passes to attention. `TitanLLaMAAttention` passes `memory_kv` through.
  `TitanLLaMAConfig` gets `num_neural_mem_kv_tokens` parameter.
- `titans_pytorch/mac_transformer.py`: `SegmentedAttention` accepts `memory_kv`
  in all three forward paths (flex, non-flex, inference), prepends to K,V, and
  updates attention masks to account for extra prefix tokens.

---

## Bug 12: Memory State Leaks Across Independent Sequences (FIXED)

**Symptom**: Neural memory retrieves memories from unrelated sequences.

**Root cause**: `memory_state` persisted across micro-batches within gradient
accumulation. With `micro_batch_size=1` and `gradient_accumulation_steps=64`,
64 independent SlimPajama sequences share accumulated memory state. Memory was
only reset every 10 optimizer steps (line 867), meaning 640 sequences contaminated
each other.

**Where it happens**:

`train_titan_llama.py` — training loop:
```python
for micro_step in range(config.gradient_accumulation_steps):
    # memory_state persists from previous micro_step!
    outputs = model(input_ids=input_ids, labels=labels, ...)
```

**Fix applied**:

1. Added `model.reset_memory_states()` before each forward pass in the
   gradient accumulation loop (train_titan_llama.py).
2. Removed the stale `if step % 10 == 0: model.reset_memory_states()` periodic
   reset, now redundant.

---

## Bug 13: `_repeat_kv_weights` Repeats Rows Instead of Heads (FIXED)

**Symptom**: Student loss ~9 even with pretrained weights correctly loaded.
Layer-by-layer comparison shows divergence starting at layer 1 (max_diff=9.25),
growing to max_diff=96 by layer 2+.

**Root cause**: `_repeat_kv_weights` used `weight.repeat_interleave(repeat_factor, dim=0)`
on the flat `(kv_heads * head_dim, hidden)` weight tensor. This repeats each **row** 4×,
not each **head** 4×.

**Where it happens**:

`titan_llama.py` — `_repeat_kv_weights()`:
```python
return weight.repeat_interleave(repeat_factor, dim=0)  # repeats ROWS, not HEADS
```

With `k_proj.weight` shape `(512, 2048)` = `(8 kv_heads × 64 head_dim, 2048)`:
- Row-level repeat: head 0 of student (rows 0–63 of repeated tensor) gets only
  the first 16 unique rows of KV head 0 (each repeated 4×), not all 64.
- Head-level repeat: head 0 of student correctly gets all 64 rows of KV head 0.

**Fix applied**:

Reshape to `(kv_heads, head_dim, hidden)`, repeat_interleave on dim=0 (head axis),
then reshape back:
```python
weight = weight.view(kv_heads, head_dim, hidden)
weight = weight.repeat_interleave(repeat_factor, dim=0)
return weight.reshape(kv_heads * repeat_factor * head_dim, hidden)
```

---

## Bug 14: Interleaved vs Split-Half RoPE Convention Mismatch (FIXED)

**Symptom**: Additional divergence from teacher even after correct weight loading.

**Root cause**: `rotary_embedding_torch` uses **interleaved** rotation: pairs
dimensions `(0,1), (2,3), (4,5), ...`. HF LLaMA uses **split-half** rotation:
pairs dimensions `(0, d/2), (1, d/2+1), (2, d/2+2), ...`. The Q,K projection
weights were trained with split-half RoPE, so applying interleaved rotation
produces wrong attention scores.

**Where it happens**:

`titans_pytorch/mac_transformer.py` — `SegmentedAttention.__init__()`:
```python
self.rotary_emb = RotaryEmbedding(dim_head, custom_freqs=rope_freqs)
```
`RotaryEmbedding.rotate_queries_with_cached_keys()` uses interleaved `rotate_half`.

**Fix applied**:

Added `apply_rotary_pos_emb_hf()` function that uses HF's split-half convention.
When `rope_freqs` are provided (from a pretrained HF model), `SegmentedAttention`
stores them as `_rope_inv_freq` and uses the HF-compatible rotation instead of
`rotary_embedding_torch`.

Also supersedes Bug 10 (RoPE theta mismatch) — by using the teacher's exact
`inv_freq` with the correct rotation convention, both the frequency values AND
the rotation pairing match the teacher perfectly.

---

## Status

Bugs 1–4, 10–14 are fixed. Bugs 5–9 are open (minor/cosmetic).

**Bugs 13+14 were the primary cause of the ~9–11 loss.** Bug 13 (KV weight repeat)
corrupted K,V projections in every layer, making attention completely wrong. Bug 14
(RoPE convention) applied rotations to wrong dimension pairs. After fixing both,
student loss matches teacher within bfloat16 rounding (delta ≈ 0.012).
