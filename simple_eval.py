import math
import time
import gc
import sys
import json
import os
import tempfile
import subprocess
from typing import Tuple, List, Dict, Optional, Any

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer
from tqdm import tqdm

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError:
    print("Warning: lm-eval not available. Install with: pip install lm-eval==0.4.2")
    evaluator = tasks = LM = None

try:
    from baseline_eval import TitanSegmentedLM
except ImportError:
    print("Warning: baseline_eval not available for advanced evaluation functions")
    TitanSegmentedLM = None


def load_small_boolq_winogrande(
    max_examples: int = 1000,
    boolq_split: str = "validation",
    winogrande_version: str = "winogrande_xl",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load small subsets of BoolQ and Winogrande as plain Python lists.

    Returns:
        boolq_data: list of {"passage", "question", "label" (bool)}
        winogrande_data: list of {"sentence", "option1", "option2", "answer" (0 or 1)}
    """
    # --- BoolQ ---
    boolq_raw = load_dataset("boolq", split=boolq_split)
    if max_examples is not None:
        boolq_raw = boolq_raw.select(range(min(max_examples, len(boolq_raw))))

    boolq_data: List[Dict] = []
    for ex in boolq_raw:
        boolq_data.append(
            {
                "passage": ex["passage"],
                "question": ex["question"],
                "label": bool(ex["answer"]),  # 1 = True, 0 = False
            }
        )

    # --- Winogrande ---
    winogrande_raw = load_dataset("winogrande", winogrande_version, split="validation")
    if max_examples is not None:
        winogrande_raw = winogrande_raw.select(
            range(min(max_examples, len(winogrande_raw)))
        )

    winogrande_data: List[Dict] = []
    for ex in winogrande_raw:
        winogrande_data.append(
            {
                "sentence": ex["sentence"],
                "option1": ex["option1"],
                "option2": ex["option2"],
                # HF stores answer as "1"/"2"
                "answer": int(ex["answer"]) - 1,  # -> 0 or 1
            }
        )

    return boolq_data, winogrande_data


def eval_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    boolq_data: List[Dict],
    device: torch.device | str = "cuda",
    batch_size: int = 8,
    max_input_len: int = 512,
) -> float:
    """
    Simple BoolQ eval using next-token scoring for ' yes' vs ' no'.
    Returns accuracy in [0, 1].

    max_input_len and batch_size should typically come from training config:
      - max_input_len ~ config.sequence_length
      - batch_size    ~ config.micro_batch_size (or smaller if OOM)
    """

    # Reset Titan memory once at the start of eval (safe even for non-Titan models)
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" no", add_special_tokens=False)[0]

    total = 0
    correct = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(boolq_data), batch_size), desc="BoolQ"):
            batch = boolq_data[i : i + batch_size]

            prompts = [
                f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer (yes or no):"
                for ex in batch
            ]
            labels = torch.tensor(
                [1 if ex["label"] else 0 for ex in batch],
                device=device,
                dtype=torch.long,
            )  # 1 = yes, 0 = no

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Extra safety if tokenizer ever returns 1D tensors
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

            # Optionally reset memory per batch to avoid cross-example bleed
            if hasattr(model, "reset_memory_states"):
                model.reset_memory_states()
            elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
                model.module.reset_memory_states()

            # print("DURING TEST: ", input_ids.shape)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits for the *next* token
            next_logits = outputs["logits"][:, -1, :]  # (batch, vocab)

            yes_logits = next_logits[:, yes_id]
            no_logits = next_logits[:, no_id]

            preds = (yes_logits > no_logits).long()  # shape (batch,)

            # Guard for shape bugs
            if preds.shape != labels.shape:
                raise RuntimeError(
                    f"BoolQ eval shape mismatch: preds {preds.shape}, labels {labels.shape}"
                )

            total += labels.numel()
            correct += (preds == labels).sum().item()

    return correct / max(1, total)


def _sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """
    Log p(text) under the model, summed over tokens.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"]  # (1, T)

    # Safety: reset memory before each sequence for Titan
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs["logits"][:, :-1, :]  # predict token t+1 from t
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, T-1, V)
        target = input_ids[:, 1:]  # shift by 1

        token_logprobs = log_probs.gather(
            2, target.unsqueeze(-1)
        ).squeeze(-1)  # (1, T-1)

        return token_logprobs.sum().item()


def eval_winogrande(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    winogrande_data: List[Dict],
    device: torch.device | str = "cuda",
    max_input_len: int = 256,
) -> float:
    """
    Winogrande eval by comparing log p(sentence with option1) vs log p(sentence with option2).
    Returns accuracy in [0, 1].

    max_input_len can be set based on training seq_len, but 256 is usually enough.
    """
    model.eval()
    device = torch.device(device)

    # Reset memory at start of eval
    if hasattr(model, "reset_memory_states"):
        model.reset_memory_states()
    elif hasattr(model, "module") and hasattr(model.module, "reset_memory_states"):
        model.module.reset_memory_states()

    total = 0
    correct = 0

    with torch.no_grad():
        for ex in tqdm(winogrande_data, desc="Winogrande"):
            sent_template = ex["sentence"]

            # Replace the blank with each option
            sent1 = sent_template.replace("_", ex["option1"])
            sent2 = sent_template.replace("_", ex["option2"])

            lp1 = _sequence_logprob(
                model, tokenizer, sent1, device, max_length=max_input_len
            )
            lp2 = _sequence_logprob(
                model, tokenizer, sent2, device, max_length=max_input_len
            )

            pred = 0 if lp1 >= lp2 else 1
            label = ex["answer"]  # 0 or 1

            total += 1
            if pred == label:
                correct += 1

    return correct / max(1, total)


def eval_winogrande_boolq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | str = "cuda",
    max_examples: int = 1000,
    *,
    # NEW: hook these up to training config
    seq_len: Optional[int] = None,
    boolq_batch_size: Optional[int] = None,
    winogrande_max_len: Optional[int] = None,
) -> dict[str, float]:
    """
    Convenience wrapper: load up to `max_examples` from BoolQ + Winogrande
    and return both accuracies.

    Pass training args in here, e.g.:

        eval_winogrande_boolq(
            model, tokenizer,
            device=device,
            max_examples=config.intermittent_eval_limit,
            seq_len=config.sequence_length,
            boolq_batch_size=config.micro_batch_size,
        )
    """
    # Sensible defaults if you *don't* pass training args
    if seq_len is None:
        seq_len = 8192  # default to your training seq len
    if boolq_batch_size is None:
        boolq_batch_size = 8
    if winogrande_max_len is None:
        # WG sentences are short; no need for full 8k
        winogrande_max_len = min(512, seq_len)

    boolq_data, winogrande_data = load_small_boolq_winogrande(
        max_examples=max_examples
    )

    print("evaluating BoolQ...")
    boolq_acc = eval_boolq(
        model,
        tokenizer,
        boolq_data,
        device=device,
        batch_size=boolq_batch_size,
        max_input_len=seq_len,
    )

    print("evaluating Winogrande...")
    winogrande_acc = eval_winogrande(
        model,
        tokenizer,
        winogrande_data,
        device=device,
        max_input_len=winogrande_max_len,
    )

    return {
        "boolq_acc": boolq_acc,
        "winogrande_acc": winogrande_acc,
    }


def quick_eval_boolq_subprocess(
    model,
    tokenizer,
    limit: int = 200,
    batch_size: int = 1,
    max_gen_toks: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Quick BoolQ evaluation in subprocess to isolate from training.
    
    Args:
        model: TitanLLaMA model
        tokenizer: Tokenizer
        limit: Number of questions to test (default 200)
        batch_size: Batch size for evaluation
        max_gen_toks: Max tokens for generation
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
        
    # Save model state to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_model_path = f.name
        
    # Save model state dict and config
    eval_model = model.module if hasattr(model, 'module') else model
    
    try:
        # Get config as dict
        config_dict = {}
        if hasattr(eval_model, 'config'):
            for key, value in eval_model.config.__dict__.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    config_dict[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to strings
                    config_dict[key] = str(value)
        
        torch.save({
            'state_dict': eval_model.state_dict(),
            'config': config_dict,
            'tokenizer_name': tokenizer.name_or_path,
        }, temp_model_path)
    except Exception as e:
        return {"error": f"Failed to save model: {str(e)}"}
    
    # Create evaluation script with extensive error logging
    eval_script = f'''
import torch
import sys
import traceback
sys.path.append("{os.getcwd()}")
import json
from transformers import AutoTokenizer

try:
    from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
    print("DEBUG: Imported TitanLLaMA classes", file=sys.stderr)
except Exception as e:
    print(f"IMPORT ERROR TitanLLaMA: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"Failed to import TitanLLaMA: {{str(e)}}"}}))
    sys.exit(1)

try:
    from baseline_eval import TitanSegmentedLM
    print("DEBUG: Imported TitanSegmentedLM", file=sys.stderr)
except Exception as e:
    print(f"IMPORT ERROR TitanSegmentedLM: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"Failed to import TitanSegmentedLM: {{str(e)}}"}}))
    sys.exit(1)

try:
    from lm_eval import evaluator, tasks
    print("DEBUG: Imported lm_eval", file=sys.stderr)
except ImportError as e:
    print(f"IMPORT ERROR lm_eval: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"lm-eval not available: {{str(e)}}"}}))
    sys.exit(1)

# Clear distributed environment
import os
for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    os.environ.pop(key, None)

try:
    # Use cuda:0 in subprocess since CUDA_VISIBLE_DEVICES maps it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device {{device}}, CUDA available: {{torch.cuda.is_available()}}", file=sys.stderr)
except Exception as e:
    print(f"DEVICE ERROR: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"Device error: {{str(e)}}"}}))
    sys.exit(1)

# Load model and tokenizer
try:
    print("DEBUG: Loading checkpoint", file=sys.stderr)
    checkpoint = torch.load("{temp_model_path}", map_location="cpu")
    print("DEBUG: Checkpoint loaded", file=sys.stderr)
    
    config = TitanLLaMAConfig(**checkpoint['config'])
    print("DEBUG: Config created", file=sys.stderr)
    
    model = TitanLLaMAForCausalLM(config)
    print("DEBUG: Model created", file=sys.stderr)
    
    model.load_state_dict(checkpoint['state_dict'])
    print("DEBUG: State dict loaded", file=sys.stderr)
    
    model = model.to(device)
    model.eval()
    print("DEBUG: Model moved to device and set to eval", file=sys.stderr)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("DEBUG: Tokenizer loaded", file=sys.stderr)
    
except Exception as e:
    print(f"MODEL LOAD ERROR: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"Failed to load model: {{str(e)}}"}}))
    sys.exit(1)

# Run evaluation
try:
    print("DEBUG: Creating TitanSegmentedLM", file=sys.stderr)
    lm = TitanSegmentedLM(
        model=model,
        tokenizer=tokenizer, 
        batch_size={batch_size},
        max_gen_toks={max_gen_toks},
    )
    print("DEBUG: TitanSegmentedLM created", file=sys.stderr)
    
    import time
    task_dict = tasks.get_task_dict(["boolq"])
    print("DEBUG: Got BoolQ task", file=sys.stderr)
    
    start_time = time.time()
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit={limit},
        bootstrap_iters=0,
    )
    eval_time = time.time() - start_time
    print("DEBUG: Evaluation completed", file=sys.stderr)
    
    boolq_results = results['results']['boolq']
    metrics = {{
        'boolq_acc': boolq_results.get('acc,none', 0.0),
        'boolq_acc_norm': boolq_results.get('acc_norm,none', 0.0),
        'eval_time_sec': eval_time,
        'questions_evaluated': {limit},
    }}
    print("DEBUG: Metrics extracted", file=sys.stderr)
    print(json.dumps(metrics))
    
except Exception as e:
    print(f"EVALUATION ERROR: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{"error": f"Evaluation failed: {{str(e)}}"}}))
    sys.exit(1)
'''

    try:
        # Write evaluation script to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(eval_script)
            script_path = f.name
            
        # Run evaluation in subprocess with proper CUDA setup
        env = os.environ.copy()
        # Make sure subprocess can see the same GPU as training
        env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
        
        # Debug: print what CUDA device we're trying to use
        print(f"DEBUG: Setting CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} for subprocess")
            
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        if result.returncode == 0:
            try:
                output = result.stdout.strip()
                if not output:
                    return {"error": "No output from subprocess"}
                return json.loads(output)
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON: {e}. STDOUT: {result.stdout[:1000]}, STDERR: {result.stderr[:1000]}"}
        else:
            return {"error": f"Subprocess failed (code {result.returncode}). FULL STDERR: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Evaluation timed out"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup temp files
        try:
            os.unlink(temp_model_path)
            os.unlink(script_path)
        except:
            pass


def quick_eval_boolq(
    model,
    tokenizer,
    limit: int = 200,
    batch_size: int = 1,
    max_gen_toks: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Quick BoolQ evaluation during training.
    
    Args:
        model: TitanLLaMA model
        tokenizer: Tokenizer
        limit: Number of questions to test (default 200)
        batch_size: Batch size for evaluation
        max_gen_toks: Max tokens for generation
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Use subprocess-based evaluation to prevent memory corruption
    return quick_eval_boolq_subprocess(
        model=model,
        tokenizer=tokenizer,
        limit=limit,
        batch_size=batch_size,
        max_gen_toks=max_gen_toks,
        device=device
    )


def quick_eval_multiple_tasks(
    model,
    tokenizer,
    tasks_to_eval: list = ["boolq", "winogrande"],
    limit: int = 100,
    batch_size: int = 1,
    max_gen_toks: int = 32,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick evaluation on multiple tasks during training.
    
    Args:
        model: TitanLLaMA model
        tokenizer: Tokenizer  
        tasks_to_eval: List of task names to evaluate
        limit: Number of examples per task
        batch_size: Batch size for evaluation
        max_gen_toks: Max tokens for generation
        device: Device to run on
    
    Returns:
        Dictionary with metrics for each task
    """
    if evaluator is None:
        return {"error": "lm-eval not available"}
    
    if device is None:
        device = next(model.parameters()).device
    
    # Unwrap DDP model if needed to avoid distributed training issues
    eval_model = model.module if hasattr(model, 'module') else model
    
    # Ensure model is on correct device
    if device is not None:
        eval_model = eval_model.to(device)
    
    # Fix distributed conflicts by clearing lm-eval distributed environment
    
    # Temporarily clear any distributed environment variables that lm-eval might have set
    orig_rank = os.environ.get('RANK')
    orig_world_size = os.environ.get('WORLD_SIZE')
    orig_local_rank = os.environ.get('LOCAL_RANK')
    orig_master_addr = os.environ.get('MASTER_ADDR')
    orig_master_port = os.environ.get('MASTER_PORT')
    
    # Force single-GPU evaluation environment
    os.environ.pop('RANK', None)
    os.environ.pop('WORLD_SIZE', None)
    os.environ.pop('LOCAL_RANK', None)
    os.environ.pop('MASTER_ADDR', None)
    os.environ.pop('MASTER_PORT', None)
    
    # Ensure CUDA context is clean
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Wrap model for LM evaluation  
    if TitanSegmentedLM is None:
        return {"error": "TitanSegmentedLM not available"}
        
    lm = TitanSegmentedLM(
        model=eval_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_gen_toks=max_gen_toks,
    )
    
    # Set model to eval mode temporarily
    was_training = model.training
    model.eval()
    
    all_metrics = {}
    
    try:
        # Evaluate each task
        for task_name in tasks_to_eval:
            try:
                task_dict = tasks.get_task_dict([task_name])
                
                start_time = time.time()
                results = evaluator.evaluate(
                    lm=lm,
                    task_dict=task_dict,
                    limit=limit,
                    bootstrap_iters=0,
                )
                eval_time = time.time() - start_time
                
                # Extract key metrics
                task_results = results['results'][task_name]
                task_metrics = {
                    f'{task_name}_acc': task_results.get('acc,none', task_results.get('acc', 0.0)),
                    f'{task_name}_acc_norm': task_results.get('acc_norm,none', 0.0),
                    f'{task_name}_eval_time': eval_time,
                }
                
                # Add additional metrics if available
                if 'f1' in task_results:
                    task_metrics[f'{task_name}_f1'] = task_results['f1']
                if 'exact_match' in task_results:
                    task_metrics[f'{task_name}_em'] = task_results['exact_match']
                    
                all_metrics.update(task_metrics)
                
            except Exception as e:
                all_metrics[f'{task_name}_error'] = str(e)
        
        # Add summary metrics
        all_metrics['total_eval_time'] = sum(
            v for k, v in all_metrics.items() if k.endswith('_eval_time')
        )
        all_metrics['questions_per_task'] = limit
        
        return all_metrics
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Restore original environment variables
        if 'orig_rank' in locals() and orig_rank is not None:
            os.environ['RANK'] = orig_rank
        if 'orig_world_size' in locals() and orig_world_size is not None:
            os.environ['WORLD_SIZE'] = orig_world_size
        if 'orig_local_rank' in locals() and orig_local_rank is not None:
            os.environ['LOCAL_RANK'] = orig_local_rank
        if 'orig_master_addr' in locals() and orig_master_addr is not None:
            os.environ['MASTER_ADDR'] = orig_master_addr
        if 'orig_master_port' in locals() and orig_master_port is not None:
            os.environ['MASTER_PORT'] = orig_master_port
            
        # Restore training mode
        if was_training:
            model.train()


def log_eval_metrics(metrics: Dict[str, Any], step: int, logger=None, wandb=None):
    """
    Log evaluation metrics to logger and wandb.
    
    Args:
        metrics: Dictionary of evaluation metrics
        step: Current training step
        logger: Logger instance (optional)
        wandb: Wandb instance (optional)
    """
    if "error" in metrics:
        if logger:
            logger.warning(f"Evaluation error at step {step}: {metrics['error']}")
        return
    
    # Log to logger
    if logger:
        log_str = f"Step {step} eval metrics: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f"{key}={value:.3f} "
        logger.info(log_str)
    
    # Log to wandb
    if wandb and hasattr(wandb, 'log'):
        wandb_metrics = {f"eval/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}
        wandb_metrics["eval/step"] = step
        wandb.log(wandb_metrics)


def should_run_intermittent_eval(step: int, eval_frequency: int = 15, start_step: int = 50) -> bool:
    """
    Determine if we should run intermittent evaluation at this step.
    
    Args:
        step: Current training step
        eval_frequency: How often to run eval (every N steps)  
        start_step: Don't start evaluating until this step
    
    Returns:
        True if should evaluate at this step
    """
    return step >= start_step and step % eval_frequency == 0
