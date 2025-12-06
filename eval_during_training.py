#!/usr/bin/env python3
"""
Lightweight evaluation during training.
Runs quick evaluations on specific tasks like BoolQ every N steps during training.
"""

import torch
import time
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
except ImportError:
    print("Warning: lm-eval not available. Install with: pip install lm-eval==0.4.2")
    evaluator = tasks = LM = None

from baseline_eval import TitanSegmentedLM


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
    if evaluator is None:
        return {"error": "lm-eval not available"}
    
    if device is None:
        device = next(model.parameters()).device
    
    # Unwrap DDP model if needed to avoid distributed training issues
    eval_model = model.module if hasattr(model, 'module') else model
    
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
    
    try:
        # Get BoolQ task
        task_dict = tasks.get_task_dict(["boolq"])
        
        # Run evaluation with time tracking
        start_time = time.time()
        results = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            bootstrap_iters=0,  # Skip confidence intervals for speed
        )
        eval_time = time.time() - start_time
        
        # Extract metrics
        boolq_results = results['results']['boolq']
        metrics = {
            'boolq_acc': boolq_results.get('acc,none', 0.0),
            'boolq_acc_norm': boolq_results.get('acc_norm,none', 0.0), 
            'eval_time_sec': eval_time,
            'questions_evaluated': limit,
        }
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # Restore training mode
        if was_training:
            model.train()


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