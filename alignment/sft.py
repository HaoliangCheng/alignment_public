#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script for MATH Dataset

This script provides a complete SFT training pipeline with:
- HuggingFace model loading with memory optimizations
- vLLM integration for evaluation with response generation
- MATH dataset evaluation with accuracy metrics
- Gradient accumulation for large effective batch sizes
- Dataset filtering and size limiting for experiments
- Enhanced wandb logging with separate train/eval metrics
- Gradient clipping with clip value 1.0
- 2 GPU setup: one for policy model, one for vLLM evaluation

Usage:
    python sft.py --model-path Qwen/Qwen2.5-Math-1.5B-Instruct --data-path data/MATH/sft.jsonl
"""

import json
import logging
import os
import torch
import torch.nn.functional as F
import typer
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import wandb
from typing import List, Dict, Optional, Callable
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import re

from helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)

from drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn,
    extract_answer,
    grade
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def filter_correct_examples(data_path: str, output_path: str, reward_fn: Callable):
    """
    Filter SFT dataset to only include examples that produce correct answers.
    
    Args:
        data_path: Path to original SFT data
        output_path: Path to save filtered data
        reward_fn: Function to evaluate correctness
        
    Returns:
        int: Number of examples in filtered dataset
    """
    logger.info(f"Filtering dataset {data_path} to only include correct examples...")
    
    correct_examples = []
    total_examples = 0
    
    with open(data_path, 'r') as f:
        for line in f:
            total_examples += 1
            example = json.loads(line)
            
            # Extract the response and ground truth
            response = example['response']
            
            # For MATH dataset, we need to extract ground truth from the original data
            # The SFT format has the solution in the response, so we need to get the original answer
            # We'll extract the answer from the <answer> tags as ground truth
            if "</think> <answer>" in response and "</answer>" in response:
                # Extract the answer that the model should produce
                gt_answer = response.split("<answer>")[-1].replace("</answer>", "").strip()
                
                # Evaluate this response against itself (since it's ground truth)
                # This filters out malformed examples
                metrics = reward_fn(response, gt_answer)
                
                # Only include if the response format is correct and answer is extractable
                if metrics["format_reward"] > 0.5 and metrics["answer_reward"] > 0.5:
                    correct_examples.append(example)
            
            if total_examples % 1000 == 0:
                logger.info(f"Processed {total_examples} examples, kept {len(correct_examples)}")
    
    # Save filtered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for example in correct_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Filtered dataset: {len(correct_examples)}/{total_examples} examples")
    return len(correct_examples)


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with size limiting and filtering."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 max_examples: Optional[int] = None, filter_correct: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                if max_examples and len(self.examples) >= max_examples:
                    break
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} training examples from {data_path}")
        if max_examples:
            logger.info(f"Limited to {max_examples} examples as requested")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract prompt and response
        prompt = example.get("prompt", example.get("problem", ""))
        response = example.get("response", example.get("answer", ""))
        
        return {
            "prompt": prompt,
            "response": response,
            "original_example": example
        }


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader that tokenizes prompts and responses."""
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    
    # Tokenize using our SFT function
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "response_mask": tokenized["response_mask"],
        "prompts": prompts,
        "responses": responses,
        "original_examples": [item["original_example"] for item in batch]
    }


def load_model_and_tokenizer(model_path: str, device: str):
    """Load HuggingFace model and tokenizer with optimizations."""
    logger.info(f"Loading model and tokenizer from {model_path}")

    if device == "auto":
        device_map = "auto"
    else:
        # Convert cuda:0 to proper device_map
        device_map = {"": device}
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        use_cache=False,  # Save memory during training
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device if not using device_map="auto"
    if device != "auto":
        model = model.to(device)
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def evaluate_model_with_vllm(
    vllm_model: LLM, 
    eval_dataset: Dataset, 
    reward_fn: Callable,
    max_eval_examples: int = 100,
    eval_batch_size: int = 16
):
    """
    Evaluate model using vLLM for generation and MATH grading for accuracy.
    
    Args:
        vllm_model: vLLM model for generation
        eval_dataset: Dataset to evaluate on
        reward_fn: Function to compute rewards/accuracy
        max_eval_examples: Maximum number of examples to evaluate
        eval_batch_size: Batch size for vLLM generation
    
    Returns:
        Dict containing evaluation metrics and examples
    """
    logger.info(f"Running evaluation on {min(max_eval_examples, len(eval_dataset))} examples")
    
    # Collect prompts and ground truths
    prompts = []
    ground_truths = []
    
    for i in range(min(max_eval_examples, len(eval_dataset))):
        example = eval_dataset[i]
        prompts.append(example["prompt"])
        
        # Extract ground truth answer from the response
        response = example["response"]  
        gt_answer = extract_answer(response)
        if gt_answer is None:
            logger.warning(f"No answer extracted from response: {response}")
            continue
        ground_truths.append(gt_answer)
    
    # Set up sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for evaluation
        top_p=1.0,
        max_tokens=512,
        stop=["</answer>"]
    )
    
    # Generate responses in batches
    all_generations = []
    for i in range(0, len(prompts), eval_batch_size):
        batch_prompts = prompts[i:i + eval_batch_size]
        batch_outputs = vllm_model.generate(batch_prompts, eval_sampling_params)
        batch_generations = [output.outputs[0].text + "</answer>" for output in batch_outputs]
        all_generations.extend(batch_generations)
    
    # Evaluate each generation
    all_metrics = []
    correct_count = 0
    format_correct_count = 0
    
    for prompt, generation, ground_truth in zip(prompts, all_generations, ground_truths):
        # Compute reward using the reward function
        metrics = reward_fn(generation, ground_truth)  
        all_metrics.append(metrics)
        
    
    # Compute aggregate metrics
    total_examples = len(all_metrics)
    
    correct_count = sum(m["reward"] > 0 for m in all_metrics)
    format_correct_count = sum(m["format_reward"] > 0 for m in all_metrics)
    avg_total_reward = sum(m["reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    avg_answer_reward = sum(m["answer_reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    avg_format_reward = sum(m["format_reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Avg Total Reward: {avg_total_reward:.3f}")
    logger.info(f"  Avg Answer Reward: {avg_answer_reward:.3f}")
    logger.info(f"  Avg Format Reward: {avg_format_reward:.3f}")
    
    return {
        "avg_total_reward": avg_total_reward,
        "avg_answer_reward": avg_answer_reward,
        "avg_format_reward": avg_format_reward,
        "correct_count": correct_count,
        "format_correct_count": format_correct_count,
        "total_examples": total_examples,
        "prompts": prompts[:10],  # Log first 10 for inspection
        "generations": all_generations[:10],
        "ground_truths": ground_truths[:10],
        "metrics": all_metrics[:10]
    }


def train_sft(
    model_path: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "sft_output",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    save_steps: int = 500,
    eval_steps: int = 100,
    logging_steps: int = 50,
    device: str = "cuda:0",
    vllm_device: str = "cuda:0",
    use_wandb: bool = False,
    wandb_project: str = "sft-training",
    max_train_examples: Optional[int] = None,
    max_eval_examples: int = 100,
    filter_correct_only: bool = False,
    gradient_clip_value: float = 1.0,
    seed: int = 42
):
    """Main SFT training function with vLLM evaluation."""
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=wandb_project, config={
            "model_path": model_path,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_length": max_length,
            "max_train_examples": max_train_examples,
            "filter_correct_only": filter_correct_only,
            "gradient_clip_value": gradient_clip_value
        })
        
        # Setup wandb metrics as specified
        wandb.define_metric("train_step")  # the x-axis for training
        wandb.define_metric("eval_step")   # the x-axis for evaluation
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle dataset filtering if requested
    actual_train_path = train_data_path
    if filter_correct_only:
        filtered_path = train_data_path.replace('.jsonl', '_filtered.jsonl')
        num_filtered = filter_correct_examples(train_data_path, filtered_path, r1_zero_reward_fn)
        actual_train_path = filtered_path
        logger.info(f"Using filtered dataset with {num_filtered} examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    model.train()
    
    # vLLM will be initialized only before final evaluation
    vllm_model = None
    
    # Create datasets
    train_dataset = SFTDataset(actual_train_path, tokenizer, max_length, max_train_examples)
    eval_dataset = SFTDataset(eval_data_path, tokenizer, max_length) if eval_data_path else None
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training for {num_epochs} epochs, {num_training_steps} total steps")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Dataset size: {len(train_dataset)} examples")
    
    # Training loop
    global_step = 0
    train_step = 0
    eval_step = 0
    running_loss = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Initialize progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Get log probabilities from model
            log_probs_result = get_response_log_probs(model, input_ids, labels)
            
            # Perform SFT microbatch training step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_probs_result["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Gradient accumulation: update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                train_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
                    
                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0]
                        })
                    
                    running_loss = 0.0
                
                # No evaluation during training - will evaluate only at the end
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                    logger.info(f"Saving checkpoint to {checkpoint_dir}")
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
    
    # Final save
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation - Initialize vLLM now and run evaluation
    if eval_dataset and eval_data_path:
        logger.info("Initializing vLLM for final evaluation...")
        
        # Initialize vLLM for final evaluation only
        if vllm_device == device:
            # Single GPU mode - use lower memory utilization
            logger.info("Using single GPU mode for evaluation")
            vllm_model = init_vllm(model_path, vllm_device, seed, gpu_memory_utilization=0.4)
        else:
            # Dual GPU mode
            logger.info("Using dual GPU mode for evaluation")
            vllm_model = init_vllm(model_path, vllm_device, seed)
        
        logger.info("Running final evaluation")
        model.eval()
        load_policy_into_vllm_instance(model, vllm_model)
        
        eval_results = evaluate_model_with_vllm(
            vllm_model, eval_dataset, r1_zero_reward_fn, max_eval_examples
        )
        
        eval_step = 1  # Single final evaluation
        
        # Compute accuracy from eval results
        final_accuracy = eval_results["correct_count"] / max(eval_results["total_examples"], 1)
        final_format_accuracy = eval_results.get("format_correct_count", 0) / max(eval_results["total_examples"], 1)
        
        if use_wandb:
            wandb.log({
                "eval/final_accuracy": final_accuracy,
                "eval/final_format_accuracy": final_format_accuracy,
                "eval/final_avg_total_reward": eval_results["avg_total_reward"],
                "eval_step": eval_step
            })
        
        # Log final generations
        log_generations(
            prompts=eval_results["prompts"],
            generations=eval_results["generations"],
            ground_truths=eval_results["ground_truths"],
            metrics=eval_results["metrics"],
            step=global_step,
            epoch=num_epochs,
            max_examples=5
        )
        
        # Clean up vLLM to free memory
        del vllm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleaned up vLLM and freed GPU memory")
    else:
        final_accuracy = None
    
    if use_wandb:
        wandb.finish()
    
    logger.info("Training completed successfully!")
    
    # Return final results for analysis
    return {
        "final_accuracy": final_accuracy,
        "model_path": output_dir,
        "train_examples": len(train_dataset)
    }


def main(
    model_path: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", help="Model name or path"),
    train_data_path: str = typer.Option("data/MATH/sft_train.jsonl", help="Training data path"),
    eval_data_path: Optional[str] = typer.Option("data/MATH/sft_val.jsonl", help="Evaluation data path"),
    output_dir: str = typer.Option("sft_output", help="Output directory"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
    num_epochs: int = typer.Option(2, help="Number of epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
    save_steps: int = typer.Option(500, help="Save checkpoint every N steps"),
    eval_steps: int = typer.Option(200, help="Evaluate every N steps"),
    logging_steps: int = typer.Option(50, help="Log every N steps"),
    device: str = typer.Option("cuda:0", help="Device for policy model"),
    vllm_device: str = typer.Option("cuda:0", help="Device for vLLM evaluation"),
    use_wandb: bool = typer.Option(True, help="Use wandb for logging"),
    wandb_project: str = typer.Option("sft-training", help="Wandb project name"),
    max_train_examples: Optional[int] = typer.Option(None, help="Maximum training examples"),
    max_eval_examples: int = typer.Option(100, help="Maximum evaluation examples"),
    filter_correct_only: bool = typer.Option(False, help="Filter to only correct examples"),
    gradient_clip_value: float = typer.Option(1.0, help="Gradient clip value"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Supervised fine-tuning training script for MATH dataset"""
    
    train_sft(
        model_path=model_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        device=device,
        vllm_device=vllm_device,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        max_train_examples=max_train_examples,
        max_eval_examples=max_eval_examples,
        filter_correct_only=filter_correct_only,
        gradient_clip_value=gradient_clip_value,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(main) 