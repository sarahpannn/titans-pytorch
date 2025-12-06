#!/usr/bin/env python3
"""
Quick eval of vanilla LLaMA without any Titan modifications.
"""

import argparse
from typing import List, Tuple, Iterable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError:
    raise SystemExit("lm-eval required: pip install lm-eval==0.4.2")


class VanillaLlamaLM(LM):
    """Simple LM wrapper for vanilla LLaMA."""
    
    def __init__(self, model, tokenizer, batch_size=1, max_gen_toks=128):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._device = next(model.parameters()).device
        # Get actual distributed environment if available
        import os
        self._rank = int(os.environ.get('RANK', 0))
        self._world_size = int(os.environ.get('WORLD_SIZE', 1))

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def loglikelihood(self, requests: Iterable) -> List[Tuple[float, bool]]:
        outputs = []
        for request in tqdm(requests):
            context, continuation = request.args if hasattr(request, "args") else request
            ll, greedy = self._loglikelihood_single(context, continuation)
            outputs.append((ll, greedy))
        return outputs

    def loglikelihood_rolling(self, requests: Iterable) -> List[Tuple[float, bool]]:
        results = []
        stride = self.max_length - 1
        for request in requests:
            (text,) = request.args if hasattr(request, "args") else request
            tokens = self.tok_encode(text)
            if not tokens:
                results.append((0.0, True))
                continue

            total_logprob = 0.0
            idx = 0
            while idx < len(tokens):
                window = tokens[max(0, idx - stride) : idx + 1]
                input_ids = torch.tensor([window], device=self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits
                logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                target_tokens = torch.tensor(window[1:], device=self.device)
                token_lp = logprobs[0, -len(target_tokens):, :].gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
                total_logprob += token_lp.sum().item()
                idx += stride

            results.append((float(total_logprob), False))
        return results

    def generate_until(self, requests: Iterable) -> List[str]:
        generations = []
        for request in tqdm(requests):
            context, gen_args = request.args if hasattr(request, "args") else request
            until = gen_args.get("until", []) if isinstance(gen_args, dict) else []
            max_new_tokens = gen_args.get("max_gen_toks", self.max_gen_toks) if isinstance(gen_args, dict) else self.max_gen_toks

            inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Handle stop tokens
            for stopper in until:
                if stopper in generated_text:
                    generated_text = generated_text[:generated_text.find(stopper)]
                    break
                    
            generations.append(generated_text)
        return generations

    def _loglikelihood_single(self, context: str, continuation: str) -> Tuple[float, bool]:
        context_ids = self.tok_encode(context)
        continuation_ids = self.tok_encode(continuation)
        
        if not context_ids:
            context_ids = [self.tokenizer.bos_token_id or self.eot_token_id]

        input_ids = torch.tensor([context_ids + continuation_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
        
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        cont_start = len(context_ids)
        cont_end = cont_start + len(continuation_ids)
        target_slice = logprobs[0, cont_start-1:cont_end-1, :]
        target_tokens = torch.tensor(continuation_ids, device=self.device)
        
        token_logprobs = target_slice.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        greedy = torch.all(target_tokens == torch.argmax(target_slice, dim=-1)).item()
        
        return float(token_logprobs.sum().item()), greedy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--tasks", nargs="+", default=["boolq"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="vanilla_eval.txt")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = VanillaLlamaLM(model, tokenizer, batch_size=args.batch_size)
    
    print(f"Running eval on {args.tasks}...")
    task_dict = tasks.get_task_dict(args.tasks)
    results = evaluator.evaluate(lm=lm, task_dict=task_dict, limit=args.limit)

    with open(args.output, "w") as f:
        f.write(str(results))
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()