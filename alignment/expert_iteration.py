#!/usr/bin/env python3
"""
Expert Iteration (EI) Training Script for MATH Dataset

This script implements Algorithm: Expert Iteration with:
- vLLM-based response sampling from current policy
- Reward-based filtering of correct responses  
- Iterative SFT training on filtered self-generated data

Algorithm: Expert iteration (EI):
1. Start with initial policy model πθ
2. For each EI step:
   a) Sample questions batch Db from dataset D
   b) Save old policy πθold ← πθ  
   c) Generate G outputs per question using πθold
   d) Compute rewards and filter correct outputs
   e) Run SFT on filtered self-generated data
   f) Update policy πθ
3. Return final policy πθ

Usage:
    python expert_iteration.py --model-path Qwen/Qwen2.5-Math-1.5B-Base --data-path data/MATH/train.jsonl
"""

import argparse
import json
import logging
import os
import random
import torch
import typer
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import wandb
from typing import List, Dict, Optional, Callable, Tuple
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import tempfile
import shutil
import math

from helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)

from drgrpo_grader import (
    r1_zero_reward_fn,  
    extract_answer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "r1_zero.prompt"
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Initialize vLLM model for sampling.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL for proper device placement
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
    Load current policy weights into vLLM instance.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class MATHDataset(Dataset):
    """Dataset for MATH problems with question sampling."""
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                if max_examples and len(self.examples) >= max_examples:
                    break
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def sample_batch(self, batch_size: int, seed: Optional[int] = None) -> List[Dict]:
        """Sample a random batch of questions."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.examples, min(batch_size, len(self.examples)))


class SFTDataset(Dataset):
    """Dataset for SFT training on generated responses."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Created SFT dataset with {len(examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "prompt": example["prompt"],
            "response": example["response"],
            "original_example": example
        }


def collate_fn(batch, tokenizer):
    """Collate function for SFT DataLoader."""
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    
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
        device_map = {"": device}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        use_cache=False,
    )
    
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    return model, tokenizer


def format_math_prompt(problem: str, prompt_template: str) -> str:
    """Format a MATH problem into the expected prompt format using r1_zero template."""
    return prompt_template.format(question=problem)


def sample_responses_with_vllm(
    vllm_model: LLM,
    questions: List[Dict],
    num_samples: int,
    prompt_template: str,
    sampling_temperature: float = 0.8,
    sampling_max_tokens: int = 512,
    sampling_min_tokens: int = 4,
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Sample multiple responses per question using vLLM.
    
    Returns:
        List of {"prompt": str, "response": str, "question_data": Dict} 
    """
    logger.info(f"Sampling {num_samples} responses for {len(questions)} questions")
    
    # Prepare prompts
    prompts = []
    question_data = []
    
    for question in questions:
        prompt = format_math_prompt(question["prompt"], prompt_template)
        prompts.append(prompt)
        question_data.append(question)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=num_samples,  # Generate num_samples responses per prompt
    )
    
    # Generate responses
    vllm_outputs = vllm_model.generate(prompts, sampling_params)
    

    # Collect all prompt-response pairs
    all_samples = []
    for i, output in enumerate(vllm_outputs):
        prompt = prompts[i]
        question = question_data[i]
        
        for generation in output.outputs:
            response = generation.text
            all_samples.append({
                "prompt": prompt,
                "response": response,
                "question_data": question,
                "logprobs": generation.logprobs if hasattr(generation, 'logprobs') else None
            })
    
    logger.info(f"Generated {len(all_samples)} total responses")
    return all_samples


def extract_ground_truth_from_solution(solution: str) -> str:
    """
    Extract the final answer from a MATH solution.
    The answer is usually in a \\boxed{} command at the end.
    """
    # Try to extract from boxed answer
    extracted = extract_answer(solution)
    if extracted:
        return extracted
    logger.warning(f"No answer extracted from solution: {solution}")
    # Fallback: return the full solution
    return solution


def filter_correct_responses(
    samples: List[Dict],
    reward_fn: Callable
) -> Tuple[List[Dict], Dict]:
    """
    Filter samples to keep only correct responses.
    
    Returns:
        Tuple of (filtered_samples, metrics)
    """
    logger.info(f"Filtering {len(samples)} samples for correctness...")

    total_samples = len(samples)
    if total_samples == 0:
        logger.warning("No samples to filter")
        return [], {
            "total_samples": 0,
            "correct_count": 0,
            "format_correct_count": 0,
            "avg_total_reward": 0.0,
            "avg_answer_reward": 0.0,
            "avg_format_reward": 0.0
        }

    metrics = {
        "total_samples": total_samples,
        "correct_count": 0,
        "format_correct_count": 0,
        "avg_total_reward": 0.0,
        "avg_answer_reward": 0.0,
        "avg_format_reward": 0.0
    }
    
    total_reward_sum = 0.0
    answer_reward_sum = 0.0
    format_reward_sum = 0.0

    correct_samples = []

    for idx, sample in enumerate(samples):
        # Extract response
        response = sample.get("response", "")
        
        # Extract ground truth answer from question data
        question = sample.get("question_data", {})
        ground_truth = None
        if "solution" in question:
            ground_truth = extract_ground_truth_from_solution(question["solution"])
        elif "response" in question:
            ground_truth = extract_ground_truth_from_solution(question["response"])
        elif "answer" in question:
            ground_truth = extract_ground_truth_from_solution(question["answer"])
        else:
            # Skip samples without ground truth
            logger.warning(f"No ground truth found for sample")
            continue
        
        # Compute reward
        try:
            reward_result = reward_fn(response, ground_truth)
        except Exception as e:
            logger.warning(f"Error computing reward for sample {idx}: {e}")
            continue

        # Track metrics for all evaluated samples
        total_reward_sum += reward_result["reward"]
        answer_reward_sum += reward_result["answer_reward"]
        format_reward_sum += reward_result["format_reward"]
        if reward_result["answer_reward"] > 0:
            correct_samples.append(sample)

    metrics["format_correct_count"] = format_reward_sum
    metrics["correct_count"] = answer_reward_sum
    metrics["avg_total_reward"] = total_reward_sum / total_samples
    metrics["avg_answer_reward"] = answer_reward_sum / total_samples
    metrics["avg_format_reward"] = format_reward_sum / total_samples

    # Final summary log
    logger.info(f"  Format correct: {metrics['format_correct_count']}/{total_samples}")
    logger.info(f"  Answer correct: {metrics['correct_count']}/{total_samples}")
    
    return correct_samples, metrics


def run_sft_on_filtered_data(
    model, 
    tokenizer,
    filtered_samples: List[Dict],
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    max_length: int = 512,
    gradient_clip_value: float = 1.0,
    device: str = "cuda:0",
    ei_step: int = 0,
    use_wandb: bool = False
) -> Dict:
    """
    Run SFT training on filtered self-generated data.
    """
    if not filtered_samples:
        logger.warning("No filtered samples for SFT training")
        return {"sft_loss": 0.0, "sft_steps": 0}
    
    logger.info(f"Running SFT on {len(filtered_samples)} filtered samples for {num_epochs} epochs")
    
    # Create SFT dataset
    sft_dataset = SFTDataset(filtered_samples, tokenizer, max_length)
    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(sft_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=max(num_training_steps // 10, 1),
        num_training_steps=max(num_training_steps, 1)
    )
    
    model.train()
    total_loss = 0.0
    total_steps = 0
    batch_losses = []  # Track losses for logging
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(sft_dataloader, desc=f"SFT Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Get log probabilities
            log_probs_result = get_response_log_probs(model, input_ids, labels)
            
            # SFT training step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_probs_result["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_steps += 1
                total_steps += 1
                
                # Log to wandb after each optimization step
                if use_wandb:
                    # Average loss over last gradient_accumulation_steps batches
                    recent_losses = batch_losses[-gradient_accumulation_steps:]
                    avg_recent_loss = sum(recent_losses) / len(recent_losses)
                    
                    wandb.log({
                        f"sft_ei_{ei_step}/loss": avg_recent_loss,
                        f"sft_ei_{ei_step}/learning_rate": scheduler.get_last_lr()[0],
                        f"sft_ei_{ei_step}/step": total_steps,
                    })
        
        avg_epoch_loss = epoch_loss / max(len(sft_dataloader), 1)
        logger.info(f"SFT Epoch {epoch + 1} completed. Avg loss: {avg_epoch_loss:.4f}")
        
        # Log epoch-level metrics
        if use_wandb:
            wandb.log({
                f"sft_ei_{ei_step}/epoch_loss": avg_epoch_loss,
                f"sft_ei_{ei_step}/epoch": epoch + 1,
            })
    
    avg_total_loss = total_loss / max(len(sft_dataloader) * num_epochs, 1)
    logger.info(f"SFT completed. Total steps: {total_steps}, Avg loss: {avg_total_loss:.4f}")
    
    return {
        "sft_loss": avg_total_loss,
        "sft_steps": total_steps,
        "sft_examples": len(filtered_samples)
    }


def evaluate_model_with_vllm(
    vllm_model: LLM,
    eval_questions: List[Dict],
    reward_fn: Callable,
    prompt_template: str,
    max_eval_examples: int = 200,
    eval_batch_size: int = 16
) -> Dict:
    """
    Evaluate current model on held-out questions with comprehensive metrics.
    Similar to sft.py implementation.
    """
    eval_questions = eval_questions[:max_eval_examples]
    logger.info(f"Evaluating model on {len(eval_questions)} questions")
    
    # Prepare prompts and ground truths
    prompts = []
    ground_truths = []
    
    for question in eval_questions:
        prompt = format_math_prompt(question["prompt"], prompt_template)
        prompts.append(prompt)
        
        # Extract ground truth from solution
        if "solution" in question:
            ground_truth = extract_ground_truth_from_solution(question["solution"])
        elif "response" in question:
            ground_truth = extract_ground_truth_from_solution(question["response"])
        elif "answer" in question:
            ground_truth = extract_ground_truth_from_solution(question["answer"])
        else:
            # Skip samples without ground truth
            logger.warning(f"No ground truth found for sample")
            continue
        ground_truths.append(ground_truth)
    
    # Set up deterministic sampling for evaluation
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
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
    
    for prompt, generation, ground_truth in zip(prompts, all_generations, ground_truths):
        # Compute reward
        metrics = reward_fn(generation, ground_truth)
        all_metrics.append(metrics)

    
    # Compute aggregate metrics (matching sft.py)
    total_examples = len(all_metrics)

    correct_count = sum(m["reward"] > 0 for m in all_metrics)
    format_correct_count = sum(m["format_reward"] > 0 for m in all_metrics)
    avg_total_reward = sum(m["reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    avg_answer_reward = sum(m["answer_reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    avg_format_reward = sum(m["format_reward"] for m in all_metrics) / total_examples if total_examples > 0 else 0.0
    
    logger.info(f"Evaluation Results for {total_examples} samples:")
    logger.info(f"  Avg Total Reward: {avg_total_reward:.3f}")
    logger.info(f"  Avg Answer Reward: {avg_answer_reward:.3f}")
    logger.info(f"  Avg Format Reward: {avg_format_reward:.3f}")

    # Return same structure as sft.py
    return {
        "avg_total_reward": avg_total_reward,
        "avg_answer_reward": avg_answer_reward,
        "avg_format_reward": avg_format_reward,
        "total_examples": total_examples,
        "correct_count": correct_count,
        "format_correct_count": format_correct_count,
        "prompts": prompts[:10],  # Log first 10 for inspection
        "generations": all_generations[:10],
        "ground_truths": ground_truths[:10],
        "metrics": all_metrics[:10]
    }


def expert_iteration(
    model_path: str,
    data_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "expert_iteration_output",
    n_ei_steps: int = 5,
    batch_size: int = 1024,
    num_rollouts: int = 4,
    sft_epochs: int = 1,
    sft_batch_size: int = 4,
    sft_gradient_accumulation_steps: int = 4,
    sft_learning_rate: float = 1e-5,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 1024,
    sampling_min_tokens: int = 4,
    max_length: int = 1024,
    gradient_clip_value: float = 1.0,
    device: str = "cuda:0",
    vllm_device: str = "cuda:0",
    use_wandb: bool = True,
    wandb_project: str = "expert-iteration",
    seed: int = 42,
    max_eval_examples: int = 100
) -> Dict:
    """
    Main expert iteration algorithm implementation.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    # Initialize wandb
    if use_wandb:
        wandb.init(project=wandb_project, config={
            "model_path": model_path,
            "n_ei_steps": n_ei_steps,
            "batch_size": batch_size,
            "num_rollouts": num_rollouts,
            "sft_epochs": sft_epochs,
            "sft_batch_size": sft_batch_size,
            "sft_learning_rate": sft_learning_rate,
            "sampling_temperature": sampling_temperature,
            "max_length": max_length
        })
        
        # Define metrics
        wandb.define_metric("ei_step")
        wandb.define_metric("ei/*", step_metric="ei_step")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompt template
    logger.info("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MATHDataset(data_path)
    eval_dataset = MATHDataset(eval_data_path) if eval_data_path else None
    
    # Load initial policy model
    logger.info("Loading initial policy model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    # Initialize vLLM
    logger.info("Initializing vLLM...")
    if vllm_device == device:
        vllm_model = init_vllm(model_path, vllm_device, seed, gpu_memory_utilization=0.4)
    else:
        vllm_model = init_vllm(model_path, vllm_device, seed)
    
    # Expert iteration loop
    results = {"ei_steps": []}
    
    for ei_step in range(n_ei_steps):
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERT ITERATION STEP {ei_step + 1}/{n_ei_steps}")
        logger.info(f"{'='*60}")
        
        # Step 1: Sample batch of questions
        logger.info(f"Sampling batch of {batch_size} questions...")
        question_batch = train_dataset.sample_batch(batch_size, seed=seed + ei_step)
        
        # Step 2: Save old policy (load current weights into vLLM)
        logger.info("Loading current policy into vLLM...")
        load_policy_into_vllm_instance(model, vllm_model)
        
        # Step 3: Sample G outputs per question
        all_samples = sample_responses_with_vllm(
            vllm_model,
            question_batch,
            num_samples=num_rollouts,
            prompt_template=prompt_template,
            sampling_temperature=sampling_temperature,
            sampling_max_tokens=sampling_max_tokens,
            sampling_min_tokens=sampling_min_tokens,
            seed=seed + ei_step
        )
        
        # Step 4: Compute rewards and filter correct outputs
        logger.info("Filtering correct responses...")
        filtered_samples, filter_metrics = filter_correct_responses(
            all_samples, r1_zero_reward_fn
        )
        
        # Step 5: Run SFT on filtered data
        sft_results = run_sft_on_filtered_data(
            model, tokenizer, filtered_samples,
            num_epochs=sft_epochs,
            batch_size=sft_batch_size,
            gradient_accumulation_steps=sft_gradient_accumulation_steps,
            learning_rate=sft_learning_rate,
            max_length=max_length,
            gradient_clip_value=gradient_clip_value,
            device=device,
            ei_step=ei_step + 1,
            use_wandb=use_wandb
        )
        
        # Step 6: Evaluate current policy
        eval_results = {}
        if eval_dataset:
            load_policy_into_vllm_instance(model, vllm_model)
            eval_results = evaluate_model_with_vllm(
                vllm_model, eval_dataset.examples, r1_zero_reward_fn, 
                prompt_template, max_eval_examples
            )
        
        # Log results
        step_results = {
            "ei_step": ei_step + 1,
            "sampled_questions": len(question_batch),
            "total_samples": len(all_samples),
            "filtered_samples": len(filtered_samples),
            "filter_rate": len(filtered_samples) / max(len(all_samples), 1),
            **filter_metrics,
            **sft_results,
            **eval_results
        }
        
        results["ei_steps"].append(step_results)

        # Log step summary
        logger.info(f"EI Step {ei_step + 1} Summary:")
        logger.info(f"  Samples: {len(all_samples)} → {len(filtered_samples)} (filter rate: {step_results['filter_rate']:.3f})")
        logger.info(f"  Format correct: {filter_metrics['format_correct_count']}/{filter_metrics['total_samples']}")
        logger.info(f"  Answer correct: {filter_metrics['correct_count']}/{filter_metrics['total_samples']} "
                   f"({filter_metrics['correct_count']/max(filter_metrics['total_samples'], 1)*100:.1f}%)")
        logger.info(f"  SFT loss: {sft_results['sft_loss']:.4f} ({sft_results['sft_steps']} steps, {sft_results.get('sft_examples', 0)} examples)")
        
        # Log SFT loss improvement if we have previous step
        if ei_step > 0:
            prev_sft_loss = results["ei_steps"][-2].get("sft_loss", 0)
            sft_loss_delta = prev_sft_loss - sft_results["sft_loss"]
            logger.info(f"  SFT loss improvement from previous step: {sft_loss_delta:+.4f}")
        
        if eval_results:
            eval_accuracy = eval_results['correct_count'] / max(eval_results['total_examples'], 1)
            logger.info(f"  Eval accuracy: {eval_accuracy:.3f} ({eval_results['correct_count']}/{eval_results['total_examples']})")
            
            # Log eval accuracy improvement if we have previous step
            if ei_step > 0 and "correct_count" in results["ei_steps"][-2]:
                prev_eval_accuracy = results["ei_steps"][-2].get("correct_count", 0) / max(results["ei_steps"][-1].get("total_examples", 1), 1)
                eval_accuracy_delta = eval_accuracy - prev_eval_accuracy
                logger.info(f"  Eval accuracy improvement from previous step: {eval_accuracy_delta:+.3f}")
        
        # Log to wandb
        if use_wandb:
            wandb_metrics = {
                "ei/step": ei_step + 1,
                "ei/avg_reward": filter_metrics["avg_total_reward"],
                "ei/avg_answer_reward": filter_metrics["avg_answer_reward"],
                "ei/avg_format_reward": filter_metrics["avg_format_reward"],
                "ei/total_samples": len(all_samples),
                "ei/filtered_samples": len(filtered_samples),
                "ei/sft_loss": sft_results["sft_loss"],
                "ei/sft_steps": sft_results["sft_steps"],
            }
            
            # Add SFT loss improvement if we have previous step data
            if ei_step > 0:
                prev_sft_loss = results["ei_steps"][-1].get("sft_loss", 0)
                current_sft_loss = sft_results["sft_loss"]
                sft_loss_delta = prev_sft_loss - current_sft_loss
                wandb_metrics["ei/sft_loss_improvement"] = sft_loss_delta
            
            # Add evaluation metrics if available
            if eval_results:
                eval_accuracy = eval_results['correct_count'] / max(eval_results['total_examples'], 1)
                wandb_metrics.update({
                    "ei/eval_avg_total_reward": eval_results["avg_total_reward"],
                    "ei/eval_avg_answer_reward": eval_results["avg_answer_reward"],
                    "ei/eval_avg_format_reward": eval_results["avg_format_reward"],
                })
                
                # Track eval accuracy improvement
                if ei_step > 0 and "correct_count" in results["ei_steps"][-2]:
                    prev_eval_accuracy = results["ei_steps"][-2].get("correct_count", 0) / max(results["ei_steps"][-1].get("total_examples", 1), 1)
                    eval_accuracy_delta = eval_accuracy - prev_eval_accuracy
                    wandb_metrics["ei/eval_accuracy_improvement"] = eval_accuracy_delta
            
            wandb.log(wandb_metrics)
        
        # Save checkpoint
        checkpoint_dir = f"{output_dir}/checkpoint_ei_step_{ei_step + 1}"
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save step results
        with open(f"{checkpoint_dir}/step_results.json", 'w') as f:
            json.dump(step_results, f, indent=2)
    
    # Final save
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save complete results
    with open(f"{output_dir}/complete_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up vLLM
    del vllm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if use_wandb:
        wandb.finish()
    
    logger.info("Expert iteration completed successfully!")
    return results


def main(
    model_path: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", help="Model path"),
    data_path: str = typer.Option("data/MATH/sft_train_filtered.jsonl", help="Training data path"),
    eval_data_path: Optional[str] = typer.Option("data/MATH/sft_val.jsonl", help="Evaluation data path"),
    output_dir: str = typer.Option("expert_iteration_output", help="Output directory"),
    n_ei_steps: int = typer.Option(5, help="Number of expert iteration steps"),
    batch_size: int = typer.Option(512, help="Batch size for question sampling (Db)"),
    num_rollouts: int = typer.Option(4, help="Number of rollouts G per question"),
    sft_epochs: int = typer.Option(1, help="Number of SFT epochs per EI step"),
    sft_batch_size: int = typer.Option(4, help="Batch size for SFT training"),
    sft_gradient_accumulation_steps: int = typer.Option(4, help="Gradient accumulation steps for SFT"),
    sft_learning_rate: float = typer.Option(1e-5, help="Learning rate for SFT"),
    sampling_temperature: float = typer.Option(1.0, help="Temperature for response sampling"),
    sampling_max_tokens: int = typer.Option(1024, help="Max tokens for response sampling"),
    sampling_min_tokens: int = typer.Option(4, help="Min tokens for response sampling"),
    max_length: int = typer.Option(1024, help="Maximum sequence length for SFT"),
    gradient_clip_value: float = typer.Option(1.0, help="Gradient clip value"),
    device: str = typer.Option("cuda:0", help="Device for policy model"),
    vllm_device: str = typer.Option("cuda:0", help="Device for vLLM sampling"),
    use_wandb: bool = typer.Option(True, help="Use wandb for logging"),
    wandb_project: str = typer.Option("expert-iteration", help="Wandb project name"),
    seed: int = typer.Option(42, help="Random seed"),
    max_eval_examples: int = typer.Option(100, help="Maximum examples for evaluation")
) -> None:
    expert_iteration(
        model_path=model_path,
        data_path=data_path,
        eval_data_path=eval_data_path,
        output_dir=output_dir,
        n_ei_steps=n_ei_steps,
        batch_size=batch_size,
        num_rollouts=num_rollouts,
        sft_epochs=sft_epochs,
        sft_batch_size=sft_batch_size,
        sft_gradient_accumulation_steps=sft_gradient_accumulation_steps,
        sft_learning_rate=sft_learning_rate,
        sampling_temperature=sampling_temperature,
        sampling_max_tokens=sampling_max_tokens,
        sampling_min_tokens=sampling_min_tokens,
        max_length=max_length,
        gradient_clip_value=gradient_clip_value,
        device=device,
        vllm_device=vllm_device,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        seed=seed,
        max_eval_examples=max_eval_examples
    )


if __name__ == "__main__":
    typer.run(main) 