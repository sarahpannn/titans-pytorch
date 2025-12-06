#!/usr/bin/env python3
"""
Lightweight evaluation during training.
Runs quick evaluations on specific tasks like BoolQ every N steps during training.
"""

import torch
import time
import gc
import sys
import json
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError:
    print("Warning: lm-eval not available. Install with: pip install lm-eval==0.4.2")
    evaluator = tasks = LM = None

from baseline_eval import TitanSegmentedLM


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
    import tempfile
    import subprocess
    import pickle
    import os
    
    if device is None:
        device = next(model.parameters()).device
        
    # Save model state to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_model_path = f.name
        
    # Save model state dict
    eval_model = model.module if hasattr(model, 'module') else model
    
    try:
        torch.save({
            'model': eval_model,  # Save the whole model object
            'tokenizer_name': tokenizer.name_or_path,
        }, temp_model_path)
    except Exception as e:
        return {"error": f"Failed to save model: {str(e)}"}
    
    # Create evaluation script
    eval_script = f'''
import torch
import sys
sys.path.append("{os.getcwd()}")
import json
from transformers import AutoTokenizer
from titan_llama import TitanLLaMAConfig, TitanLLaMAForCausalLM
from baseline_eval import TitanSegmentedLM

try:
    from lm_eval import evaluator, tasks
except ImportError:
    print(json.dumps({{"error": "lm-eval not available"}}))
    sys.exit(0)

# Clear distributed environment
import os
for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    os.environ.pop(key, None)

device = torch.device("{device}")

# Load model and tokenizer
try:
    checkpoint = torch.load("{temp_model_path}", map_location="cpu")
    model = checkpoint['model'].to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(json.dumps({{"error": f"Failed to load model: {{str(e)}}"}}))
    sys.exit(1)

# Run evaluation
lm = TitanSegmentedLM(
    model=model,
    tokenizer=tokenizer, 
    batch_size={batch_size},
    max_gen_toks={max_gen_toks},
)

try:
    import time
    task_dict = tasks.get_task_dict(["boolq"])
    start_time = time.time()
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit={limit},
        bootstrap_iters=0,
    )
    eval_time = time.time() - start_time
    
    boolq_results = results['results']['boolq']
    metrics = {{
        'boolq_acc': boolq_results.get('acc,none', 0.0),
        'boolq_acc_norm': boolq_results.get('acc_norm,none', 0.0),
        'eval_time_sec': eval_time,
        'questions_evaluated': {limit},
    }}
    print(json.dumps(metrics))
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''

    try:
        # Write evaluation script to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write(eval_script)
            script_path = f.name
            
        # Run evaluation in subprocess
        env = os.environ.copy()
        if hasattr(device, 'index'):
            env['CUDA_VISIBLE_DEVICES'] = str(device.index)
        elif 'cuda' in str(device):
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
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
    import os
    
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


# Example usage function to add to training loop
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