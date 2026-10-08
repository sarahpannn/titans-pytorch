from __future__ import annotations
from typing import Callable

import math
import os
from functools import partial
from itertools import zip_longest
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from assoc_scan import AssocScan

from titans_pytorch.memory_models import(
    MemoryMLP,
    ResidualNorm
)

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

LinearNoBias = partial(Linear, bias = False)

# neural mem state related

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
    # A masked store can end with an incomplete chunk.  Its mask must be cached
    # beside the tokens; otherwise the next one-token decode concatenates (say)
    # 511 cached tokens with one new token but still has only a length-1 mask.
    'cache_store_mask',
], defaults=(None,))

def detach_tree(value):
    """Detach every tensor in a nest of tuples/lists, leaving other values be."""
    if is_tensor(value):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    return NeuralMemState(*detach_tree(tuple(state)))

# functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

# softclamping gradients

def softclamp_max(t, max_value):
    # Smoothly approach ``max_value`` without imposing a non-zero floor.
    # The previous shifted-tanh formula mapped 0 -> max_value / 2, which made
    # zero gradients become NaN when the caller divided by their zero norm.
    return (t / max_value).tanh() * max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t

    t, inverse = pack_one_with_inverse(t, 'bn *')

    # Use at least fp32 for low-precision inner gradients, but preserve fp64
    # when checking the recurrence with numerical derivatives.
    norm_input = t if t.dtype == torch.float64 else t.float()
    norm = norm_input.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    # For small norms, tanh(x) ~= x, so this preserves the gradient. For large
    # norms it smoothly caps at max_value. A clamped denominator makes an
    # exactly-zero gradient remain exactly zero instead of 0 * inf -> NaN.
    scale = clamped_norm / norm.clamp_min(torch.finfo(norm.dtype).tiny)
    t = t * scale.to(dtype = t.dtype)
    return inverse(t)

# spectral norming the surprise update w/ newton schulz matrix iter
# Keller Jordan et al. from OSS w/ nanogpt, now being used for two works, Atlas and 'TTT done right'

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)

# multi head rmsnorm

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# chunk pooling

class AveragePool(Module):
    def __init__(
        self,
        chunk_size
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim = -2)

        return reduce(x * attn, 'b n c d -> b n d', 'sum')

# main neural memory

def gelu_grad(t):
    """Derivative of the exact (erf) GELU used by MemoryMLP."""
    cdf = 0.5 * (1. + torch.erf(t * (2. ** -0.5)))
    pdf = torch.exp(-0.5 * t * t) * (1. / math.sqrt(2. * math.pi))
    return cdf + t * pdf

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)



class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        batch_size = None,
        dim_head = None,
        heads = 1,
        retrieve_heads = None,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        per_parameter_lr_modulation = False, # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation = 1., # max of 10.
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1, # whether a single token can do multiple updates to the memory model
        qkv_receives_diff_views = True, # to address an issue raised by a phd student (who will be credited if experiments are green). basically the issue raised is that the memory MLP is only learning Wk @ Wv linear mapping and that may not be expressive enough. we will use hyper connections to allow the network to choose different previous layer inputs as keys / values and see if that does anything
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        detach_inner_grads: bool = True,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        spectral_norm_surprises = False,
        gated_transition = False,
        mem_model_norm_add_residual = True, # by default, layernorm output and add residual as proposed in TTT paper, but could be removed
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        ),
        bypass_projections: bool = False, # if True, to_queries/to_keys/to_values and the pre-norms become Identity; caller must supply pre-projected tensors of shape (B, N, heads*dim_head)
        heterogeneous_qkv: bool = False, # fixed input contract: Q may be wider than K/V and is supplied as a (Q, K, V) tuple
        history_context_dim: int | None = None,
        history_mix_init: float = 0.1,
    ):
        super().__init__()

        # Expensive, synchronization-heavy diagnostics intended for short
        # correctness runs.  When enabled, abort at the first non-finite tensor
        # inside the fast-weight store/retrieve path instead of letting the
        # failure surface later as a generic non-finite outer gradient.
        self.fail_on_nonfinite = os.environ.get("NMM_FAIL_ON_NONFINITE", "0") == "1"


        # experimental separate-delta retrieval flags

        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)
        retrieve_heads = default(retrieve_heads, heads)
        assert divisible_by(retrieve_heads, heads), (
            f'retrieve heads ({retrieve_heads}) must be divisible by memory heads ({heads})'
        )
        assert retrieve_heads == heads or bypass_projections, (
            'distinct retrieve heads require bypass_projections=True'
        )
        assert retrieve_heads == heads or not (post_rmsnorm or qk_rmsnorm), (
            'distinct retrieve heads do not support multi-head QK/post RMSNorm'
        )
        assert True, (
            'distinct retrieve heads do not support separate-delta retrieval'
        )
        self.bypass_projections = bypass_projections

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # batch size

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)

        self.batch_size = batch_size

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        # key values receiving different views

        self.qkv_receives_diff_views = qkv_receives_diff_views
        if heterogeneous_qkv and not qkv_receives_diff_views:
            raise ValueError('heterogeneous_qkv requires qkv_receives_diff_views=True')
        self.heterogeneous_qkv = heterogeneous_qkv

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads
        self.retrieve_heads = retrieve_heads
        self.retrieve_heads_per_memory_head = retrieve_heads // heads
        self.dim_head = dim_head

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_retrieve_heads = Rearrange(
            'b n (h d) -> b h n d', h = retrieve_heads
        )
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = (
            LinearNoBias(dim_inner, dim)
            if heads > 1 and retrieve_heads == heads
            else nn.Identity()
        )

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 and retrieve_heads == heads else None

        if exists(history_context_dim):
            if not 0. < history_mix_init < 1.:
                raise ValueError('history_mix_init must lie strictly between 0 and 1')
            mix_logit = math.log(history_mix_init / (1. - history_mix_init))
            self.history_mix = Parameter(torch.full((retrieve_heads, dim_head), mix_logit))
            self.history_mix_proj = Linear(history_context_dim, retrieve_heads * dim_head, bias = False)
            nn.init.zeros_(self.history_mix_proj.weight)
        else:
            self.register_parameter('history_mix', None)
            self.history_mix_proj = None

        # memory model

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        # validate memory model

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')

            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

        # the memory is the weights of the model

        if mem_model_norm_add_residual:
            model = ResidualNorm(dim = dim_head, model = model)

        # The store / retrieve path runs the fast-weight network with
        # explicit matmuls and closed-form gradients rather than autograd, so it
        # is specialised to a plain MemoryMLP: a chain of weight matrices with
        # GELU between them, no biases and no buffers. Other memory models would
        # need their own gradient equations.
        if not isinstance(model, MemoryMLP):
            raise NotImplementedError(
                'the fixed dense fast-weight layout supports a plain MemoryMLP '
                f'only, got {type(model).__name__}; note that '
                'mem_model_norm_add_residual=True wraps the model in ResidualNorm'
            )

        self.memory_model = model
        self.memory_activation = model.activation

        mem_model_params = dict(model.named_parameters())

        self.num_memory_parameter_tensors = len(mem_model_params)

        memory_model_parameters = [*mem_model_params.values()]

        if per_head_learned_parameters:
            memory_model_parameters = [repeat(p, '... -> h ...', h = heads) for p in memory_model_parameters]

        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        assert num_kv_per_token > 0

        self.to_keys = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.to_values = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        # Tied-mode bypass: caller supplies pre-projected K/V/Q at dim_inner
        # (== heads * dim_head). Identity out the pre-norm and projections so
        # no trainable mapping is applied on entry.
        if bypass_projections:
            assert dim == dim_inner * num_kv_per_token, (
                f"bypass_projections requires dim ({dim}) == dim_inner*num_kv_per_token "
                f"({dim_inner}*{num_kv_per_token}); the caller feeds pre-projected tensors"
            )
            self.retrieve_norm = nn.Identity()
            self.store_norm = nn.Identity()
            self.to_queries = nn.Identity()
            self.to_keys = nn.Identity()
            self.to_values = nn.Identity()

        self.store_memory_loss_fn = store_memory_loss_fn

        self.num_kv_per_token = num_kv_per_token

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # momentum related

        self.to_momentum = Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange('b n (h o) -> o (b h) n', h = heads),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # learned weight residual

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm
        self.detach_inner_grads = detach_inner_grads

        # spectral norming the surprises before update, a la Muon from Jordan et al.

        self.spectral_norm_surprises = spectral_norm_surprises

        # weight decay factor

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # learned transition, as seeing instability when decreasing neural mem batch size
        # perhaps it can slowly learn to adjust from early residual to fully transitioning to new weights every batch size

        self.transition_gate = nn.Parameter(tensor(-5.)) if gated_transition else None

        # inits

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # CUDA-graph-safe decode (opt-in; default keeps the original path).
        # The partial chunk lives in a fixed-size buffer here rather than in a
        # NeuralMemState tensor that grows one token per step. See
        # _graph_safe_accumulate.
        self.graph_safe_decode = False
        # Escape hatch for the sub-chunk store skip, so tests can run the
        # original path and diff against it.
        self._skip_empty_store = True
        self.register_buffer('_gs_store_buffer', None, persistent = False)
        self.register_buffer('_gs_write_index', None, persistent = False)
        # Set by the caller *outside* the compiled region: True on the token that
        # completes a store chunk. It must be a bool, not a counter -- dynamo
        # guards on the exact value of an int attribute it reads, so a 0..chunk-1
        # counter compiles one graph per slot. Two values, two graphs.
        self._gs_will_flush = False
        # Host-side mirror of the write index, for the eager bounds check only.
        self._gs_host_slot = 0

    def reset_graph_safe_store(self):
        """Drop the graph-safe partial-chunk buffer. Call between sequences."""
        self._gs_store_buffer = None
        self._gs_write_index = None
        self._gs_will_flush = False
        self._gs_host_slot = 0

    def _graph_safe_accumulate(self, store_seq):
        """Buffer one decode token; return a full chunk when the boundary lands.

        Returns ``None`` while the chunk is still filling, which tells the caller
        to leave the memory state untouched -- exactly what the original
        ``num_chunks == 0`` early return did.

        The write index is a device tensor so every accumulate step replays the
        same graph (a Python index would bake one slot per step, needing
        ``chunk_size`` graphs). The accumulate-vs-flush decision is a plain bool
        set by the caller before the compiled region is entered: dynamo guards on
        it and specializes into exactly two graphs. Deriving it from a counter
        *inside* this function does not work -- dynamo guards on the counter's
        exact value, not on the comparison, so that recompiles once per slot.
        """
        chunk_size = self.store_chunk_size

        if self._gs_store_buffer is None:
            shape = list(store_seq.shape)
            shape[-2] = chunk_size
            self._gs_store_buffer = torch.zeros(
                shape, device = store_seq.device, dtype = store_seq.dtype
            )
            self._gs_write_index = torch.zeros(
                1, device = store_seq.device, dtype = torch.long
            )
            if not torch._dynamo.is_compiling():
                torch._dynamo.mark_static_address(self._gs_store_buffer)
                torch._dynamo.mark_static_address(self._gs_write_index)

        token_dim = store_seq.ndim - 2
        if not torch._dynamo.is_compiling():
            # Eager only, and against a host-side mirror rather than the device
            # tensor: reading _gs_write_index would sync once per layer per
            # token (measured at ~1.8 ms/token). dynamo never traces this branch,
            # so the mirror cannot become a guard and recompile per slot.
            if self._gs_host_slot >= chunk_size:
                raise RuntimeError(
                    f"graph-safe store buffer overflow: slot "
                    f"{self._gs_host_slot} >= chunk_size {chunk_size}. The "
                    "caller must set _gs_will_flush on the token completing "
                    "each chunk."
                )
            self._gs_host_slot = 0 if self._gs_will_flush else self._gs_host_slot + 1
        self._gs_store_buffer.index_copy_(token_dim, self._gs_write_index, store_seq)
        self._gs_write_index += 1

        if not self._gs_will_flush:
            return None

        self._gs_write_index.zero_()
        return self._gs_store_buffer

    # --- explicit fast-weight network ------------------------------------
    #
    # Fast weights are carried as a fixed-length tuple of dense tensors, one per
    # MemoryMLP matrix, laid out as [BH, D_in, D_out] (or [BH, N, D_in, D_out]
    # when a per-chunk timeline is present). Both methods below broadcast over
    # every leading dimension, so the same code serves the flat and timeline
    # forms without any shape inference.

    @staticmethod
    def _split_dense_state(flat, shapes, sizes, lead):
        """Split a packed fast-weight buffer back into per-matrix tensors.

        ``flat`` carries ``lead`` leading dimensions followed by one feature axis
        holding every parameter tensor end to end.
        """
        out = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            piece = flat[..., offset:offset + size]
            out.append(piece.reshape(*piece.shape[:lead], *shape))
            offset += size
        return tuple(out)

    def memory_forward(self, weights, x):
        """Run the configured fast-weight MLP with no biases."""
        hidden = x
        for index, weight in enumerate(weights):
            if index > 0:
                hidden = F.gelu(hidden)
            hidden = hidden @ weight
        return hidden

    def memory_grads(self, weights, keys, loss_weights, values):
        """Closed-form gradient of the weighted store loss w.r.t. each matrix.

        Replaces ``vmap(grad(functional_call(...)))``. The loss is
        ``sum_t w_t * mean_d (M(k_t) - v_t)^2`` per sample, matching eq. (12);
        the returned auxiliary loss is the *unweighted* per-token term, exactly
        as the autograd version reported it.
        """
        # Forward, retaining each matmul's input and each pre-activation.
        layer_inputs = []
        pre_activations = []
        hidden = keys
        for index, weight in enumerate(weights):
            if index > 0:
                pre_activations.append(hidden)
                hidden = F.gelu(hidden)
            layer_inputs.append(hidden)
            hidden = hidden @ weight

        prediction = hidden
        loss = self.store_memory_loss_fn(prediction, values)

        # d/d_pred of sum_t w_t * mean_d (pred - v)^2
        grad_output = (
            2. * loss_weights.unsqueeze(-1) * (prediction - values) / values.shape[-1]
        )

        grads = [None] * len(weights)
        for index in range(len(weights) - 1, -1, -1):
            grads[index] = layer_inputs[index].transpose(-1, -2) @ grad_output
            if index > 0:
                grad_output = grad_output @ weights[index].transpose(-1, -2)
                grad_output = grad_output * gelu_grad(
                    pre_activations[index - 1]
                )

        return tuple(grads), loss

    def _check_finite(self, tensor_value: Tensor, name: str):
        if not self.fail_on_nonfinite or not is_tensor(tensor_value) or tensor_value.numel() == 0:
            return

        detached = tensor_value.detach()
        finite_mask = torch.isfinite(detached)
        if bool(finite_mask.all()):
            return

        nan_count = int(torch.isnan(detached).sum().item())
        posinf_count = int(torch.isposinf(detached).sum().item())
        neginf_count = int(torch.isneginf(detached).sum().item())
        finite_values = detached.float()[finite_mask]
        finite_range = "no finite values"
        if finite_values.numel() > 0:
            finite_range = (
                f"finite_min={finite_values.min().item():.6e}, "
                f"finite_max={finite_values.max().item():.6e}"
            )
        raise FloatingPointError(
            f"NMM non-finite tensor at {name}: shape={tuple(detached.shape)}, "
            f"dtype={detached.dtype}, nan={nan_count}, +inf={posinf_count}, "
            f"-inf={neginf_count}, {finite_range}"
        )

    def _check_finite_weights(self, tensors, name: str):
        if not self.fail_on_nonfinite:
            return
        for index, tensor_value in enumerate(tensors):
            self._check_finite(tensor_value, f"{name}.{index}")

    def init_weights(
        self,
        batch,
    ):
        if self.per_head_learned_parameters:
            return tuple(
                repeat(parameter, 'h ... -> (b h) ...', b = batch)
                for parameter in self.memory_model_parameters
            )

        return tuple(
            repeat(parameter, '... -> bh ...', bh = batch * self.heads)
            for parameter in self.memory_model_parameters
        )

    def init_momentum(
        self,
        batch,
    ):
        return tuple(
            weight.new_zeros((self.momentum_order, *weight.shape))
            for weight in self.init_weights(batch)
        )

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True
    ):
        self._check_finite(seq, "store.input")
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        # shapes and variables

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]

        # The remainder is carried in the returned state as `cache_store_segment`
        # and outlives this call. As a view it pins `seq`'s entire storage -- the
        # whole prefill -- in order to keep at most chunk_size-1 rows, and when
        # seq_len is a multiple of chunk_size it pins the lot to keep nothing at
        # all. Measured at 128K over 16 layers: 4.0 GiB held by tensors of shape
        # (2, 1, 0, 512). .contiguous() would not help; a contiguous slice keeps
        # its base storage. The copy is bounded by chunk_size, so it is cheap.
        remainder = remainder.clone()

        next_seq_len_index = seq_index + round_down_seq_len

        # init weights if needed
        # weights of the memory network

        if not exists(weights):
            weights = self.init_weights(batch)

        weights = tuple(weights)
        self._check_finite_weights(weights, "store.initial_weights")

        # allow for neural memory of a previous layer to influence surprise of current layer

        weights_for_surprise = tuple(
            repeat(weight, 'bh ... -> bh n ...', n = num_chunks) for weight in weights
        )

        # initial norm

        seq = self.store_norm(seq)
        self._check_finite(seq, "store.normalized_input")

        # handle keys and values coming from different sequences from hyper connection

        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)
        self._check_finite(adaptive_lr, "store.adaptive_lr")

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()
        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # keys and values

        keys = self.to_keys(seq)
        values = self.to_values(values_seq)
        self._check_finite(keys, "store.keys_before_head_split")
        self._check_finite(values, "store.targets_before_head_split")

        # maybe multi head

        keys, values = map(self.split_kv_heads, (keys, values))

        # maybe keys rmsnorm

        keys = self.k_norm(keys)

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates) for t in (keys, values))

        # adaptive lr

        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # optionally a storing memories mask can be passed in. if False, will set the learning rate to 0. for those positions

        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)

            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # maybe add previous layer weight

        assert xnor(exists(self.to_learned_weight_residual_mix), exists(prev_weights))

        if exists(prev_weights):

            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks

            prev_weights = tuple(t[:, start_index:end_index] for t in prev_weights)

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights = tuple(
                    einx.multiply('bh n, bh n ... -> bh n ...', mix, t) for t in prev_weights
                )

            weights_for_surprise = tuple(
                surprise_weight + prev_weight
                for surprise_weight, prev_weight in zip(weights_for_surprise, prev_weights)
            )

        # flatten batch and time if surprise depends on previous layer memory model

        weights_for_surprise = tuple(
            rearrange(weight, 'bh n ... -> (bh n) ...') for weight in weights_for_surprise
        )
        self._check_finite_weights(weights_for_surprise, "store.weights_for_surprise")

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        # First-order outer training detaches the ingredients of the inner
        # surprise gradient, but deliberately keeps the learned adaptive step
        # live.  This makes eta trainable from the outer loss without building
        # second-order gradients through M(K), K, V, or the fast weights.
        #
        # Previously eta was multiplied into the surprise and then the completed
        # surprise tensor was detached below.  Its parameters consequently sat
        # in the optimizer with requires_grad=True but received no gradient.
        if self.detach_inner_grads:
            surprise_weights = detach_tree(weights_for_surprise)
            surprise_keys = keys.detach()
            surprise_values = values.detach()
        else:
            surprise_weights = weights_for_surprise
            surprise_keys = keys
            surprise_values = values

        # Audit-only inference controls.  Production and training behavior stays
        # exactly on the original one-step descent path unless a diagnostic
        # explicitly sets these runtime attributes.
        audit_inner_steps = int(getattr(self, "audit_inner_steps", 1))
        audit_inner_update_sign = float(
            getattr(self, "audit_inner_update_sign", -1.0)
        )
        if audit_inner_steps < 1:
            raise ValueError("audit_inner_steps must be at least 1")
        if audit_inner_update_sign not in (-1.0, 1.0):
            raise ValueError("audit_inner_update_sign must be -1 (descent) or +1 (ascent)")

        grads, unweighted_mem_model_loss = self.memory_grads(
            surprise_weights, surprise_keys, adaptive_lr, surprise_values
        )

        self._check_finite(unweighted_mem_model_loss, "store.inner_loss")
        self._check_finite_weights(grads, "store.inner_grad_before_clamp")

        if self.detach_inner_grads:
            # The surprise inputs were detached before differentiation above, so
            # the only live path through `grads` is the learned adaptive step.
            # Momentum, decay, and optional modulation are applied later and also
            # remain differentiable.
            unweighted_mem_model_loss = unweighted_mem_model_loss.detach()

        flat_adaptive_lr = adaptive_lr

        def prepare_inner_grads(raw_grads):
            if exists(self.max_grad_norm):
                raw_grads = tuple(
                    softclamp_grad_norm(t, self.max_grad_norm) for t in raw_grads
                )
            self._check_finite_weights(raw_grads, "store.inner_grad_after_clamp")
            prepared = tuple(
                rearrange(t, '(bh n) ... -> bh n ...', bh = batch * heads)
                for t in raw_grads
            )
            if need_layer_lr_mod:
                prepared = tuple(
                    einx.multiply('b h, b h ... -> b h ...', modulation, t)
                    for modulation, t in zip(layer_lr_mod, prepared)
                )
            return prepared

        grads = prepare_inner_grads(grads)

        # The normal path is deliberately kept as the same single negated
        # gradient expression. Diagnostics can instead repeatedly optimize the
        # same chunk, or flip the sign to gradient ascent.
        audit_records = getattr(self, "_audit_inner_loss_records", None)
        if audit_inner_steps == 1:
            # Preserve the ordinary one-step expression exactly.
            surprises = (
                tuple(-t for t in grads)
                if audit_inner_update_sign == -1.0
                else tuple(t for t in grads)
            )
            if audit_records is not None:
                final_inner_weights = tuple(
                    weight + rearrange(delta, 'bh n ... -> (bh n) ...')
                    for weight, delta in zip(surprise_weights, surprises)
                )
        else:
            step_weights = surprise_weights
            for step_index in range(audit_inner_steps):
                if step_index > 0:
                    step_raw_grads, _ = self.memory_grads(
                        step_weights, surprise_keys, flat_adaptive_lr,
                        surprise_values,
                    )
                    self._check_finite_weights(
                        step_raw_grads, f"store.inner_grad_step_{step_index + 1}"
                    )
                    step_grads = prepare_inner_grads(step_raw_grads)
                else:
                    step_grads = grads
                step_weights = tuple(
                    weight + audit_inner_update_sign * rearrange(
                        grad, 'bh n ... -> (bh n) ...'
                    )
                    for weight, grad in zip(step_weights, step_grads)
                )
            final_inner_weights = step_weights
            surprises = tuple(
                rearrange(final - initial, '(bh n) ... -> bh n ...', bh = batch * heads)
                for final, initial in zip(final_inner_weights, surprise_weights)
            )

        # Capture the objective change only when an audit opts in. This extra
        # forward is absent from ordinary training and inference.
        if audit_records is not None:
            final_prediction = self.memory_forward(final_inner_weights, surprise_keys)
            final_unweighted_loss = self.store_memory_loss_fn(
                final_prediction, surprise_values
            )
            self._check_finite(final_unweighted_loss, "store.inner_loss_after")
            eta_sum = flat_adaptive_lr.float().sum().clamp_min(1e-12)
            audit_records.append({
                "steps": audit_inner_steps,
                "update_sign": audit_inner_update_sign,
                "unweighted_before": float(unweighted_mem_model_loss.float().mean()),
                "unweighted_after": float(final_unweighted_loss.float().mean()),
                "eta_weighted_before": float(
                    (unweighted_mem_model_loss.float() * flat_adaptive_lr.float()).sum()
                    / eta_sum
                ),
                "eta_weighted_after": float(
                    (final_unweighted_loss.float() * flat_adaptive_lr.float()).sum()
                    / eta_sum
                ),
            })

        # surprises

        adaptive_lr = rearrange(flat_adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

        self._check_finite_weights(surprises, "store.surprises")

        # past states

        if not exists(past_state):
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper

            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)

            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early return if sequence length less than chunk size

        if num_chunks == 0:
            updates = tuple(weight.unsqueeze(1) for weight in weights)
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)

            output = (updates, next_store_state)

            if not return_surprises:
                return output

            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates
        #
        # The fast-weight state is one dense buffer here: every parameter tensor
        # is flattened and concatenated so that the momentum and decay
        # recurrences each run as a single scan over the whole state, rather than
        # one scan per parameter tensor. The gates broadcast over the trailing
        # feature axis, so this is elementwise-identical to scanning separately.

        parameter_shapes = [tuple(surprise.shape[2:]) for surprise in surprises]
        parameter_sizes = [math.prod(shape) for shape in parameter_shapes]

        update = cat([surprise.flatten(2) for surprise in surprises], dim = -1)
        last_update = cat([weight.flatten(1) for weight in past_last_update], dim = -1)

        # derive momentum with associative scan - eq (10)

        if has_momentum:
            momentum = update

            momentums = [] # stores all momentum orders starting with first, to generalize to Nth order momentum

            last_momentum = cat(
                [moment.flatten(2) for moment in past_last_momentum], dim = -1
            )

            # go from first order momentum all the way to the Nth

            for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum) # momentum is S / surprise in the paper

                momentums.append(momentum)

            momentums = stack(momentums)

            next_last_momentum = self._split_dense_state(
                momentums[:, :, -1], parameter_shapes, parameter_sizes, lead = 2,
            )

            if learned_combine and self.learned_combine_include_zeroth:
                # add the original surprise if learned combination of momentums
                momentums = cat((rearrange(update, '... -> 1 ...'), momentums), dim = 0)

            if not learned_combine:
                update = momentums[-1]
            else:
                update = einsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')
        else:
            next_last_momentum = tuple(past_last_momentum)

        # maybe spectral norm surprises
        #
        # Newton-Schulz acts on each weight matrix, so the dense buffer is split
        # for the duration of that step and re-packed for the decay scan.

        if self.spectral_norm_surprises:
            update = cat([
                newtonschulz5(one_update).flatten(2)
                for one_update in self._split_dense_state(
                    update, parameter_shapes, parameter_sizes, lead = 2,
                )
            ], dim = -1)

        # Decay only the episodic delta around the immutable learned slow map.
        # If W_t = W_slow + Delta_t, the desired recurrence is
        #
        #   W_t = W_slow + (1 - decay_t) * Delta_(t-1) + surprise_t
        #       = (1 - decay_t) * W_(t-1)
        #         + decay_t * W_slow + surprise_t.
        #
        # The previous implementation omitted the anchor term and therefore
        # decayed the complete map toward zero, erasing pretrained query
        # dependence even when the write surprise was disabled.
        slow_weight = cat(
            [weight.flatten(1) for weight in self.init_weights(batch)], dim = -1
        )
        update = update + decay_factor * slow_weight.unsqueeze(1)
        update = self.assoc_scan(
            1. - decay_factor, update, prev = last_update, remove_prev = False
        )
        self._check_finite(update, "store.fast_weight_updates")

        updates = self._split_dense_state(
            update, parameter_shapes, parameter_sizes, lead = 2,
        )
        next_last_update = tuple(one_update[:, -1] for one_update in updates)

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        compressed_updates = tuple(update[:, -1:].contiguous() for update in updates)

        next_store_state = NeuralMemState(
            next_seq_len_index,
            weights,
            remainder,
            next_state,
            compressed_updates,
        )

        # next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        # return updates to neural memory at all chunked timesteps + neural mem cache / state to be fed back

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: tuple[Tensor, ...],
        chunk_alignment: str = "token_shift",
    ):
        """Read queries against a per-chunk fast-weight timeline.

        ``weights`` is always in timeline form -- a tuple of dense tensors shaped
        ``[batch * heads, num_weight_chunks, D_in, D_out]``. Callers normalise to
        that shape, which is what lets this method drop the previous runtime
        inspection of weight shapes.
        """
        self._check_finite(seq, "retrieve.input")
        self._check_finite_weights(weights, "retrieve.fast_weights")
        if chunk_alignment not in {"token_shift", "inclusive"}:
            raise ValueError(f"unknown retrieval chunk_alignment: {chunk_alignment}")

        chunk_size = self.retrieve_chunk_size
        batch, seq_len = seq.shape[:2]
        retrieve_heads = self.retrieve_heads
        retrieve_groups = self.retrieve_heads_per_memory_head

        num_weight_steps = weights[0].shape[1]

        # a single token read against a single set of weights is a decode step

        if seq_len == 1 and num_weight_steps == 1:
            chunk_size = 1

        # padding related, for chunked processing

        next_seq_len = round_up_multiple(seq_len, chunk_size)

        padding = next_seq_len - seq_len
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # pre norm

        seq = self.retrieve_norm(seq)
        self._check_finite(seq, "retrieve.normalized_input")

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_retrieve_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)
        self._check_finite(queries, "retrieve.queries")

        # A single KV-head memory may be queried by multiple attention Q heads.
        # Duplicate only the *view* of each memory's weights, so all Q heads in
        # a GQA group run the same fast-weight MLP and accumulate gradients
        # back into that shared KV-head state.

        if retrieve_groups > 1:
            weights = tuple(
                rearrange(
                    repeat(
                        rearrange(weight, '(b h) n ... -> b h n ...', b = batch, h = self.heads),
                        'b h n ... -> b h g n ...',
                        g = retrieve_groups,
                    ),
                    'b h g n ... -> (b h g) n ...',
                )
                for weight in weights
            )

        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

        # align the weight timeline to the query chunks

        num_query_steps = queries.shape[0] // (batch * retrieve_heads)

        if num_weight_steps != num_query_steps:
            weights = self._align_weight_steps(
                weights,
                batch * retrieve_heads,
                num_weight_steps,
                num_query_steps,
                chunk_alignment,
            )

        weights = tuple(
            rearrange(weight, 'bh n ... -> (bh n) ...') for weight in weights
        )

        # run the fast-weight network

        values = self.memory_forward(weights, queries)
        self._check_finite(values, "retrieve.memory_model_output")

        # reconstitute batch dimension

        values = rearrange(
            values,
            '(b h n) c d -> b h (n c) d',
            b = batch,
            h = retrieve_heads,
        )

        values = self.multihead_rmsnorm(values)
        self._check_finite(values, "retrieve.post_rmsnorm_output")

        # maybe gate

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine

        values = self.merge_heads(values)

        values = self.combine_heads(values)
        self._check_finite(values, "retrieve.final_output")

        return values[:, :seq_len]

    @staticmethod
    def _align_weight_steps(
        weights,
        num_rows,
        num_weight_steps,
        num_query_steps,
        chunk_alignment,
    ):
        """Match a weight timeline of one length to a query timeline of another.

        ``store_memories`` returns ``[theta_0, ..., theta_W]``. ``token_shift`` is
        causal -- window ``w`` reads ``theta_w``; ``inclusive`` instead reads
        ``theta_(w+1)``. A timeline shorter than the queries is held at its last
        entry.
        """
        if num_weight_steps > num_query_steps:
            start = 0 if chunk_alignment == "token_shift" else 1
            return tuple(
                weight[:, start:start + num_query_steps] for weight in weights
            )

        if num_weight_steps == num_query_steps:
            return weights

        pad_steps = num_query_steps - num_weight_steps
        return tuple(
            cat((weight, repeat(weight[:, -1:], 'bh 1 ... -> bh p ...', p = pad_steps)), dim = 1)
            for weight in weights
        )

    def forward_sequence(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None,
        retrieve_chunk_alignment: str = "token_shift",
    ):
        is_multi_input = self.qkv_receives_diff_views

        # The QKV representation is fixed when the module is constructed. This
        # keeps the hot path monomorphic even when GQA makes Q wider than K/V.
        if self.heterogeneous_qkv:
            assert len(seq) == 3, 'heterogeneous QKV input must contain (Q, K, V)'
            retrieve_seq, key_seq, value_seq = seq
            seq = stack((key_seq, value_seq))

        # handle single token

        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, '... b d -> ... b 1 d')

        is_single_token = seq.shape[-2] == 1

        # if different views for qkv, then

        if is_multi_input and not self.heterogeneous_qkv:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = retrieve_seq if self.heterogeneous_qkv else seq

        # handle previous state init

        if not exists(state):
            state = NeuralMemState(0, None, None, None, None)

        # default the store sequence to the (value-view) input sequence

        store_seq = default(store_seq, seq)

        # A one-token decode reads theta_(t-1), then writes K_t -> V_t. Longer
        # sequences build the update timeline first; token_shift aligns each
        # query with the preceding chunk's weights.

        if is_single_token:
            retrieve_weights = state.updates
            if not exists(retrieve_weights):
                # no timeline yet: read against W0, shaped as a 1-step timeline
                retrieve_weights = tuple(
                    weight.unsqueeze(1)
                    for weight in self.init_weights(retrieve_seq.shape[0])
                )
            else:
                retrieve_weights = tuple(
                    update[:, -1:].contiguous() for update in retrieve_weights
                )
            retrieved = self.retrieve_memories(retrieve_seq, retrieve_weights)

        next_neural_mem_state, surprises = self._store_sequence(
            store_seq,
            state,
            store_mask = store_mask,
            prev_weights = prev_weights,
            ttt_batch_size = ttt_batch_size,
            is_single_token = is_single_token,
        )

        if not is_single_token:
            retrieved = self.retrieve_memories(
                retrieve_seq,
                next_neural_mem_state.updates,
                chunk_alignment=retrieve_chunk_alignment,
            )

        # maybe detach

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: NeuralMemState | None = None,
        context: Tensor | None = None,
        store_mask: Tensor | None = None,
        detach_mem_state: bool = False,
        seq_offset: int = 0,
    ) -> tuple[Tensor, Tensor | None, NeuralMemState]:
        """Read Q from strictly-past ``M(K) -> V`` weights, then update them.

        Q uses query heads while K/V use memory heads. Inputs stay in attention
        head layout so the model-facing path has one stable tensor contract.
        """
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError('projected Q/K/V must have shape (batch, heads, tokens, dim_head)')
        if k.shape != v.shape:
            raise ValueError(f'projected K and V shapes must match, got {k.shape} and {v.shape}')
        if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError(f'projected Q and K token shapes must match, got {q.shape} and {k.shape}')
        if q.shape[1] != self.retrieve_heads or k.shape[1] != self.heads:
            raise ValueError(
                f'expected {self.retrieve_heads} Q heads and {self.heads} KV heads, '
                f'got {q.shape[1]} and {k.shape[1]}'
            )

        batch, _, tokens, dim_head = q.shape
        if dim_head != self.dim_head:
            raise ValueError(f'expected head dimension {self.dim_head}, got {dim_head}')

        address_scale = dim_head ** -0.25

        def flatten_heads(t: Tensor) -> Tensor:
            return rearrange(t, 'b h n d -> b n (h d)')

        values, next_state = self.forward_sequence(
            (
                flatten_heads(q * address_scale),
                flatten_heads(k * address_scale),
                flatten_heads(v),
            ),
            state = state,
            store_mask = store_mask,
            detach_mem_state = detach_mem_state,
            retrieve_chunk_alignment = 'token_shift',
        )

        values = rearrange(
            values,
            'b n (h d) -> b n h d',
            h = self.retrieve_heads,
            d = self.dim_head,
        )

        history_mix = None
        if exists(self.history_mix_proj):
            if not exists(context):
                raise ValueError('context is required when history mixing is enabled')
            history_mix_logits = (
                rearrange(
                    self.history_mix_proj(context),
                    'b n (h d) -> b n h d',
                    h = self.retrieve_heads,
                    d = self.dim_head,
                )
                + self.history_mix.view(1, 1, self.retrieve_heads, self.dim_head)
            )
            history_mix = history_mix_logits.sigmoid()

        # There is no content-addressed memory before the first complete TTT
        # store chunk. ``forward_sequence`` necessarily uses W0 to preserve a
        # stable tensor shape when no update timeline exists, but exposing that
        # initialization as a memory *read* changes the backbone from token 0
        # and can dominate short-prompt generation. The first query allowed to
        # read is the token immediately after positions [0, chunk_size) have
        # produced their completed update. Mask the blend as well as the value:
        # lerping toward a zero value with a nonzero history coefficient would
        # still attenuate local attention before any memory exists.
        positions = torch.arange(
            tokens, device=values.device, dtype=torch.long
        ) + torch.as_tensor(seq_offset, device=values.device, dtype=torch.long)
        has_completed_ttt_update = (positions >= self.store_chunk_size).view(
            1, tokens, 1, 1
        )
        values = values * has_completed_ttt_update.to(values.dtype)
        if exists(history_mix):
            history_mix = history_mix * has_completed_ttt_update.to(history_mix.dtype)

        return rearrange(values, 'b n h d -> b n (h d)'), history_mix, next_state

    @torch.no_grad()
    def initialize_kv_memory(
        self,
        key_projection: Tensor,
        value_projection: Tensor,
        identity: bool = False,
    ) -> None:
        """Initialize each memory-head MLP in the projected K/V address space."""
        params = self.memory_model_parameters
        depth = len(params)
        if depth < 2 or any(param.ndim != 3 for param in params):
            raise ValueError('projected K/V initialization requires a per-head MLP of depth >= 2')

        for index, parameter in enumerate(params):
            params[index] = Parameter(parameter.detach().clone())

        square = (self.dim_head, self.dim_head)
        if any(param.shape[-2:] != square for param in params):
            raise ValueError('projected K/V initialization requires square memory matrices')

        key_projection = key_projection.reshape(self.heads, self.dim_head, -1)
        value_projection = value_projection.reshape(self.heads, self.dim_head, -1)
        final = params[-1]
        eye = torch.eye(self.dim_head, device=final.device, dtype=final.dtype)
        address_scale = self.dim_head ** -0.25
        gelu_gain = 2. ** (depth - 1)

        for parameter in params[:-1]:
            parameter.copy_(eye.unsqueeze(0).expand_as(parameter))
        if identity:
            final.copy_((gelu_gain * eye).unsqueeze(0).expand_as(final))
            return

        for head in range(self.heads):
            design = (address_scale * key_projection[head].transpose(0, 1)).float()
            target = value_projection[head].transpose(0, 1).float()
            linear_map = torch.linalg.lstsq(design, target).solution
            final[head].copy_((gelu_gain * linear_map).to(final.dtype))


    def _store_sequence(
        self,
        store_seq,
        state: NeuralMemState,
        store_mask: Tensor | None = None,
        prev_weights = None,
        ttt_batch_size: int | None = None,
        is_single_token: bool = False,
    ):
        """Run the store half of `forward()`: chunked inner-loop weight updates
        over `store_seq`. Returns ``(next_state, surprises)``.

        ``next_state.updates`` holds the full per-chunk cumulative update timeline.

        Shared by ``forward()`` (online) and ``store_only()`` (offline build) so
        the two paths can never drift apart.
        """
        seq_index, weights, cache_store_seq, past_state, _ = state[:5]
        cache_store_mask = state.cache_store_mask

        # take care of cache
        #
        # Graph-safe decode keeps the partial chunk in a fixed-size module buffer
        # instead of a state tensor that grows one token per step. The state
        # tensor is both a CUDA-graph hazard (produced inside the capture, read
        # on the next replay) and a shape that cycles 1..chunk_size, which would
        # need one graph per slot. Semantics are unchanged: tokens accumulate in
        # order and the store runs on exactly the same chunk boundaries.
        if self.graph_safe_decode and is_single_token:
            if exists(store_mask):
                raise RuntimeError(
                    "graph_safe_decode does not yet support masked neural-memory stores"
                )
            # The module buffer starts empty, so decode must begin on a chunk
            # boundary or the buffer and the leftover state disagree about which
            # tokens are pending. Prefill lengths are multiples of chunk_size in
            # practice; refuse rather than silently drop the remainder.
            if exists(cache_store_seq) and cache_store_seq.shape[-2] > 0:
                raise RuntimeError(
                    "graph_safe_decode requires decode to start on a store-chunk "
                    f"boundary, but {cache_store_seq.shape[-2]} tokens are still "
                    f"pending (store_chunk_size={self.store_chunk_size})"
                )
            flushed = self._graph_safe_accumulate(store_seq)
            if flushed is None:
                # Below the chunk boundary: the original code would take the
                # num_chunks == 0 early return, leaving weights, momentum and the
                # update timeline untouched and advancing seq_index by 0.
                return state, (None, None)
            store_seq = flushed
            cache_store_seq = None

        if exists(cache_store_seq):
            cached_tokens = cache_store_seq.shape[-2]
            store_seq = safe_cat((cache_store_seq, store_seq))
            if exists(store_mask):
                if not exists(cache_store_mask):
                    # A missing historical mask means those tokens came through
                    # the ordinary unmasked store path and therefore were valid.
                    cache_store_mask = torch.ones(
                        (*store_mask.shape[:-1], cached_tokens),
                        dtype=torch.bool,
                        device=store_mask.device,
                    )
                store_mask = torch.cat((cache_store_mask, store_mask), dim=-1)
            elif exists(cache_store_mask):
                current_tokens = store_seq.shape[-2] - cached_tokens
                current_mask = torch.ones(
                    (*cache_store_mask.shape[:-1], current_tokens),
                    dtype=torch.bool,
                    device=cache_store_mask.device,
                )
                store_mask = torch.cat((cache_store_mask, current_mask), dim=-1)

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, default(ttt_batch_size, self.batch_size)

        need_update_weights = exists(batch_size)

        # Below a chunk boundary the store cannot produce an update, but
        # store_memories still builds the weights tuple, repeats it for n = 0
        # chunks and normalizes an empty tensor before bailing out at
        # num_chunks == 0. At decode that is 511 wasted calls in every 512, and
        # the store is the largest single item in a Titan layer's step
        # (0.536 ms/token of 1.765 ms measured at 16K). Produce that early
        # return's state directly instead.
        #
        # Two conditions, both necessary. The chunk test is why no update can
        # happen; the batch test is because a TTT batch boundary rebases
        # `weights` onto the last update (see update_after_final_store below)
        # even when no chunk completed.
        if (
            self._skip_empty_store
            and store_seq_len < chunk_size
            and exists(weights)
            and exists(past_state)
            and not (need_update_weights and divisible_by(seq_index + store_seq_len, batch_size))
        ):
            source = past_state[0] if is_single_token else weights
            skipped_state = NeuralMemState(
                seq_index,
                weights,
                store_seq,
                past_state,
                tuple(one_weight.unsqueeze(1) for one_weight in source),
                store_mask,
            )
            return skipped_state, (None, None)

        # determine split sizes and when to update

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)

            # Boundaries fall where the absolute position ``seq_index + offset``
            # is a multiple of ``batch_size``. This is integer arithmetic on
            # three Python ints; computing it with tensors instead forced a
            # data-dependent branch that broke every torch.compile graph here.
            first_boundary = (-(seq_index + 1)) % batch_size + 1

            indices = [0, *range(first_boundary, store_seq_len + 1, batch_size)]

            if indices[-1] != store_seq_len:
                indices.append(store_seq_len)

            split_sizes = [
                end - start for start, end in zip(indices[:-1], indices[1:])
            ]

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        # accumulate updates

        updates = None

        def accum_updates(past_updates, future_updates):
            if not exists(past_updates):
                return future_updates
            if not exists(future_updates):
                return past_updates

            return tuple(
                cat((past_update[:, :-1], future_update), dim = 1)
                for past_update, future_update in zip(past_updates, future_updates)
            )

        # loop through chunks of store sequences

        store_seqs = store_seq.split(split_sizes, dim = -2)

        if exists(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)

        # whether to allow network to slowly adjust from initial weight throughout (residual path) to fully updating weights every batch

        surprises = (None, None)
        gate = None

        if exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index=seq_index,
                past_state=past_state,
                prev_weights=prev_weights,
                mask=maybe_store_mask,
                return_surprises=True,
            )
            remainder_len = next_neural_mem_state.cache_store_segment.shape[-2]
            remainder_mask = None
            if exists(maybe_store_mask):
                remainder_mask = maybe_store_mask[..., -remainder_len:].clone()
                if remainder_len == 0:
                    remainder_mask = maybe_store_mask[..., :0].clone()
            next_neural_mem_state = next_neural_mem_state._replace(
                cache_store_mask=remainder_mask
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states

            updates = accum_updates(updates, next_updates)
            surprises = tuple(safe_cat(args, dim=-1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = tuple(
                    one_weight.lerp(one_last_update, gate)
                    for one_weight, one_last_update in zip(weights, last_update)
                )

            past_state = (last_update, last_momentum)
            weights = last_update

            next_neural_mem_state = next_neural_mem_state._replace(
                weights=weights,
                states=past_state,
            )

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = tuple(update.unsqueeze(1) for update in last_update)

        next_neural_mem_state = next_neural_mem_state._replace(updates=updates)

        return next_neural_mem_state, surprises

    def store_only(
        self,
        store_seq,
        state: NeuralMemState | None = None,
        store_mask: Tensor | None = None,
        detach_mem_state: bool = False,
        ttt_batch_size: int | None = None,
        prev_weights = None,
    ) -> NeuralMemState:
        """Consume memory tokens and update the NeuralMemState, producing no
        retrieved memory vectors (offline memory build).

        For `qkv_receives_diff_views` (tied) inputs `store_seq` is the stacked
        ``(queries, keys, values)`` view; the query view is dropped here since
        only keys/values drive the inner-loop store. See `read_only` for the
        retrieval half.
        """
        is_multi_input = self.qkv_receives_diff_views

        if self.heterogeneous_qkv:
            assert len(store_seq) == 3, 'heterogeneous QKV input must contain (Q, K, V)'
            store_seq = stack((store_seq[1], store_seq[2]))

        if store_seq.ndim == 2 or (is_multi_input and store_seq.ndim == 3):
            store_seq = rearrange(store_seq, '... b d -> ... b 1 d')

        is_single_token = store_seq.shape[-2] == 1

        if is_multi_input and not self.heterogeneous_qkv:
            # drop the retrieve (query) view; keep the key / value views for storing
            store_seq = store_seq[1:]

        if not exists(state):
            state = NeuralMemState(0, None, None, None, None)

        next_neural_mem_state, _ = self._store_sequence(
            store_seq,
            state,
            store_mask = store_mask,
            prev_weights = prev_weights,
            ttt_batch_size = ttt_batch_size,
            is_single_token = is_single_token,
        )

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        return next_neural_mem_state

    def read_only(
        self,
        retrieve_seq,
        state: NeuralMemState,
    ) -> tuple[Tensor, NeuralMemState]:
        """Retrieve for query tokens against an already-built NMM `state`, without
        updating it from the query tokens (offline read-only retrieval).

        Every query position reads against the fully-built (final) memory weights,
        so the offline memory is uniformly visible to all query tokens. `state` is
        returned unchanged. ``store_memories`` is never called on `retrieve_seq`.
        """
        assert exists(state) and exists(state.updates), (
            'read_only requires a state whose updates were already built; call store_only first'
        )

        is_multi_input = self.qkv_receives_diff_views

        if self.heterogeneous_qkv:
            assert len(retrieve_seq) == 3, 'heterogeneous QKV input must contain (Q, K, V)'
            retrieve_seq = retrieve_seq[0]

        # A heterogeneous Q view is already tokenized as (B, N, H_q * D).
        # The legacy stacked multi-view form is (3, B, D) and still needs a
        # singleton token dimension before selecting its query view below.
        if retrieve_seq.ndim == 2 or (
            is_multi_input and retrieve_seq.ndim == 3 and not self.heterogeneous_qkv
        ):
            retrieve_seq = rearrange(retrieve_seq, '... b d -> ... b 1 d')

        if is_multi_input and not self.heterogeneous_qkv:
            # use only the query view for retrieval
            retrieve_seq = retrieve_seq[0]

        # collapse the update timeline to the final fully-built memory weights so
        # all query positions retrieve against the same complete memory
        final_update = tuple(
            update[:, -1:].contiguous() for update in state.updates
        )

        retrieved = self.retrieve_memories(retrieve_seq, final_update)

        return retrieved, state
