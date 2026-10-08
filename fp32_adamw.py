"""AdamW with persistent FP32 master weights for low-precision models."""

from contextlib import contextmanager

import torch
from torch.optim import AdamW


class FP32AdamW(AdamW):
    """Keep model/gradient buffers unchanged outside the optimizer step.

    Master weights and moments live in optimizer state, so ordinary checkpoint
    saves preserve updates smaller than a BF16 ULP. Legacy AdamW checkpoints
    start their master weights from the loaded model and promote their moments.
    Call after backward and gradient clipping; closures are not supported.
    """

    @contextmanager
    def _master_parameters(self, *, loading=False):
        swapped = []
        try:
            for group in self.param_groups:
                for parameter in group['params']:
                    if parameter.dtype not in (torch.bfloat16, torch.float16):
                        continue
                    if not loading and parameter.grad is None:
                        continue
                    data, gradient = parameter.data, parameter.grad
                    master = self.state.get(parameter, {}).get('fp32_master')
                    if master is None:
                        master = data.float().clone()
                    parameter.grad = None
                    parameter.data = master
                    if gradient is not None:
                        parameter.grad = gradient.float()
                    swapped.append((parameter, data, gradient))
            yield swapped
        finally:
            for parameter, data, gradient in swapped:
                parameter.grad = None
                parameter.data = data
                parameter.grad = gradient

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('FP32AdamW does not support closures')
        with self._master_parameters() as swapped:
            result = super().step()
            for parameter, data, _ in swapped:
                self.state[parameter]['fp32_master'] = parameter.data
                data.copy_(parameter.data)
        return result

    def load_state_dict(self, state_dict):
        # AdamW casts loaded moments to parameter dtype. Temporarily exposing
        # FP32 parameters prevents it from quantizing saved FP32 state to BF16.
        with self._master_parameters(loading=True):
            super().load_state_dict(state_dict)
