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

## Status

Both bugs are now fixed. The student model should start with hidden states matching
the teacher at initialization (before neural memory contributions), allowing the
distillation loss to start low and the LM loss to reflect the pretrained model's
actual capability.
