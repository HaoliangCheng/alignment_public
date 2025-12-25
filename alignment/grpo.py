#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training Loop

This script implements the complete GRPO algorithm as described in Section 7.1,
using the helper functions for reward computation and loss calculation.
"""

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
import typer
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Import our helper functions
from helper import (
    tokenize_prompt_and_output,
    get_response_log_probs
)
from helper_rl import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from drgrpo_grader import (
    r1_zero_reward_fn, 
    extract_answer
)
from expert_iteration import (
    load_policy_into_vllm_instance,
    extract_ground_truth_from_solution
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MATHDataset(Dataset):
    """Dataset for MATH problems"""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
        
def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "r1_zero.prompt"
    with open(prompt_path, 'r') as f:
        return f.read().strip()

def create_r1_zero_prompt(problem: str, prompt_template: str) -> str:
    """Create the r1_zero prompt format for MATH dataset using the template"""
    return prompt_template.format(question=problem)


def compute_reward(response: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute reward for a response given ground truth using r1_zero_reward_fn.
    Returns dict with keys: reward, format_reward, answer_reward
    """
    # Use the robust reward function from drgrpo_grader
    return r1_zero_reward_fn(response, ground_truth, fast=True)


def generate_responses(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
) -> List[str]:
    """Generate responses using vLLM"""
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def compute_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    requires_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities for generated responses
    Returns: (log_probs, response_mask) both shape (batch_size, max_seq_len)
    
    Args:
        requires_grad: If False, wraps computation in torch.no_grad() for efficiency
    """
    # Use helper function to tokenize prompts and responses
    tokenized = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    
    # Set model to appropriate mode for gradient computation
    if requires_grad:
        model.train()
    else:
        model.eval()
    
    # Use helper function to get log probabilities
    log_prob_result = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=False,
    )
    
    response_log_probs = log_prob_result["log_probs"].to(device)
    
    return response_log_probs, response_mask


def train_grpo(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    dataset_path: str = "data/MATH/train.jsonl",
    val_dataset_path: str = "data/MATH/test.jsonl",
    output_dir: str = "grpo_outputs",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.35,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    gradient_clip_value: float = 1.0,
    validation_frequency: int = 5,
    validation_size: int = 1024,
    log_frequency: int = 1,
    seed: int = 42,
    wandb_project: Optional[str] = None,
) -> None:
    """
    Main GRPO training function
    """
    
    # Validate hyperparameters
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if specified
    if wandb_project:
        wandb.init(
            project=wandb_project,
            config={
                "model_name": model_name,
                "n_grpo_steps": n_grpo_steps,
                "learning_rate": learning_rate,
                "rollout_batch_size": rollout_batch_size,
                "group_size": group_size,
                "loss_type": loss_type,
                "use_std_normalization": use_std_normalization,
                "cliprange": cliprange,
            }
        )
    
    # Load prompt template
    logger.info("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt()
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = MATHDataset(dataset_path, "train")
    val_dataset = MATHDataset(val_dataset_path, "test")
    
    # Initialize models
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Policy model for training
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=False,
    )
    
    # vLLM for generation
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="float16",
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        top_p=1.0,
        stop=["</answer>"],  # Stop at end of answer tag
        include_stop_str_in_output=True,  # Include stop string in output for reward checking
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting GRPO training...")
    
    for step in range(n_grpo_steps):
        step_start_time = time.time()
        
        # Sample prompts for this rollout batch
        prompt_indices = random.sample(range(len(train_dataset)), n_prompts_per_rollout_batch)
        sampled_problems = [train_dataset[i] for i in prompt_indices]
        prompts = [create_r1_zero_prompt(problem["prompt"], prompt_template) for problem in sampled_problems]
        # Extract ground truth answers (not full response with <think> tags)
        ground_truths = [extract_ground_truth_from_solution(problem["response"]) for problem in sampled_problems]
        
        # Repeat prompts and ground truths for group_size
        repeated_prompts = []
        repeated_ground_truths = []
        for prompt, gt in zip(prompts, ground_truths):
            repeated_prompts.extend([prompt] * group_size)
            repeated_ground_truths.extend([gt] * group_size)
        
        # Update vLLM with current policy weights (on-policy sampling)
        logger.info(f"Step {step}: Loading current policy weights into vLLM...")
        load_policy_into_vllm_instance(policy, llm)
        
        # Generate responses
        logger.info(f"Step {step}: Generating {rollout_batch_size} responses...")
        responses = generate_responses(llm, repeated_prompts, sampling_params)
        
        # Compute rewards and advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=compute_reward,
            rollout_responses=responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        
        # Compute old log probabilities if needed for off-policy
        old_log_probs = None
        if loss_type == "grpo_clip":
            old_log_probs, _ = compute_log_probs(
                policy, tokenizer, repeated_prompts, responses, device, requires_grad=False
            )

        
        # Training epochs on the rollout batch
        total_loss = 0.0
        total_grad_norm = 0.0
        n_updates = 0
        
        for epoch in range(epochs_per_rollout_batch):
            # Shuffle data for this epoch
            indices = list(range(rollout_batch_size))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            epoch_updates = 0
            
            # Process microbatches
            for mb_start in range(0, rollout_batch_size, micro_train_batch_size):
                mb_end = min(mb_start + micro_train_batch_size, rollout_batch_size)
                mb_indices = indices[mb_start:mb_end]
                
                # Get microbatch data
                mb_prompts = [repeated_prompts[i] for i in mb_indices]
                mb_responses = [responses[i] for i in mb_indices]

                mb_advantages = advantages[mb_indices]
                mb_raw_rewards = raw_rewards[mb_indices]
                mb_advantages = mb_advantages.to(device)
                mb_raw_rewards = mb_raw_rewards.to(device)
                # Compute current log probabilities
                curr_log_probs, response_mask = compute_log_probs(
                    policy, tokenizer, mb_prompts, mb_responses, device
                )
                
                # Prepare arguments for grpo_microbatch_train_step
                train_step_kwargs = {
                    "policy_log_probs": curr_log_probs,
                    "response_mask": response_mask,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "loss_type": loss_type,
                }
                
                if loss_type == "no_baseline":
                    train_step_kwargs["raw_rewards"] = mb_raw_rewards.unsqueeze(1)
                elif loss_type == "reinforce_with_baseline":
                    train_step_kwargs["advantages"] = mb_advantages.unsqueeze(1)
                elif loss_type == "grpo_clip":
                    train_step_kwargs["advantages"] = mb_advantages.unsqueeze(1)
                    train_step_kwargs["old_log_probs"] = old_log_probs[mb_indices]
                    train_step_kwargs["cliprange"] = cliprange
                
                # Use helper function for training step
                scaled_loss, loss_metadata = grpo_microbatch_train_step(**train_step_kwargs)
                
                epoch_loss += scaled_loss
                epoch_updates += 1
                
                # Gradient accumulation step
                if (epoch_updates % gradient_accumulation_steps == 0) or (mb_end == rollout_batch_size):
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), gradient_clip_value
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_grad_norm += grad_norm.item()
                    n_updates += 1
            
            total_loss += epoch_loss / epoch_updates if epoch_updates > 0 else 0.0
        
        # Log training metrics
        step_time = time.time() - step_start_time
        
        log_dict = {
            "step": step,
            "step_time": step_time,
            "train_reward_total": reward_metadata["raw_reward_mean"],
            "train_reward_format": reward_metadata["format_reward_mean"],
            "train_reward_answer": reward_metadata["answer_reward_mean"],
            "advantage_mean": reward_metadata["advantage_mean"],
            "advantage_std": reward_metadata["advantage_std"],
        }
        
        # Add loss-specific metrics
        if loss_type == "grpo_clip" and "clip_fraction" in loss_metadata:
            log_dict["clip_fraction"] = loss_metadata["clip_fraction"].item()
        
        if step % log_frequency == 0:
            logger.info(f"Step {step}: reward={reward_metadata['raw_reward_mean']:.4f}")
        
        # Validation
        if step % validation_frequency == 0 and step > 0:
            val_rewards = run_validation(
                llm, val_dataset, sampling_params, validation_size, prompt_template
            )
            log_dict.update({
                "val_reward_total": val_rewards["total"],
                "val_reward_format": val_rewards["format"],
                "val_reward_answer": val_rewards["answer"],
            })
            logger.info(f"Validation rewards - Total: {val_rewards['total']:.4f}, Format: {val_rewards['format']:.4f}, Answer: {val_rewards['answer']:.4f}")
        
        # Log to wandb
        if wandb_project:
            wandb.log(log_dict, step=step)
        
        # Save checkpoint
        if step % 50 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}")
            policy.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(output_dir, "final_model")
    policy.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")
    
    # Clean up vLLM (similar to expert_iteration.py)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if wandb_project:
        wandb.finish()


def run_validation(
    llm: LLM,
    val_dataset: MATHDataset,
    sampling_params: SamplingParams,
    validation_size: int,
    prompt_template: str,
) -> Dict[str, float]:
    """Run validation and compute reward metrics"""
    
    # Sample validation problems
    val_indices = random.sample(range(len(val_dataset)), min(validation_size, len(val_dataset)))
    val_problems = [val_dataset[i] for i in val_indices]
    
    # Create prompts
    val_prompts = [create_r1_zero_prompt(problem["prompt"], prompt_template) for problem in val_problems]
    # Extract ground truth answers (not full response with <think> tags)
    val_ground_truths = [extract_ground_truth_from_solution(problem["response"]) for problem in val_problems]
    
    # Generate responses
    val_responses = generate_responses(llm, val_prompts, sampling_params)
    
    # Compute rewards
    total_rewards = []
    format_rewards = []
    answer_rewards = []
    
    for response, gt in zip(val_responses, val_ground_truths):
        rewards = compute_reward(response, gt)
        total_rewards.append(rewards["reward"])
        format_rewards.append(rewards["format_reward"])
        answer_rewards.append(rewards["answer_reward"])
    
    return {
        "total": sum(total_rewards) / len(total_rewards),
        "format": sum(format_rewards) / len(format_rewards),
        "answer": sum(answer_rewards) / len(answer_rewards),
    }


def main(
    model_name: str = typer.Option("Qwen/Qwen2.5-Math-1.5B", help="Model name or path"),
    dataset_path: str = typer.Option("data/MATH/sft_train_filtered.jsonl", help="Training dataset path"),
    val_dataset_path: str = typer.Option("data/MATH/sft_val.jsonl", help="Validation dataset path"),
    output_dir: str = typer.Option("grpo_outputs", help="Output directory"),
    n_grpo_steps: int = typer.Option(200, help="Number of GRPO steps"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
    advantage_eps: float = typer.Option(1e-6, help="Advantage epsilon"),
    rollout_batch_size: int = typer.Option(256, help="Rollout batch size"),
    group_size: int = typer.Option(8, help="Group size"),
    sampling_temperature: float = typer.Option(1.0, help="Sampling temperature"),
    sampling_min_tokens: int = typer.Option(4, help="Minimum tokens"),
    sampling_max_tokens: int = typer.Option(1024, help="Maximum tokens"),
    epochs_per_rollout_batch: int = typer.Option(1, help="Epochs per rollout batch"),
    train_batch_size: int = typer.Option(256, help="Training batch size"),
    gradient_accumulation_steps: int = typer.Option(128, help="Gradient accumulation steps"),
    gpu_memory_utilization: float = typer.Option(0.35, help="GPU memory utilization"),
    loss_type: str = typer.Option("reinforce_with_baseline", help="Loss type with choices: grpo_clip/no_baseline/reinforce_with_baseline"),
    use_std_normalization: bool = typer.Option(True, help="Use std normalization"),
    cliprange: float = typer.Option(0.2, help="Clip range for GRPO"),
    gradient_clip_value: float = typer.Option(1.0, help="Gradient clip value"),
    validation_frequency: int = typer.Option(5, help="Validation frequency"),
    validation_size: int = typer.Option(1024, help="Validation size"),
    log_frequency: int = typer.Option(1, help="Log frequency"),
    seed: int = typer.Option(42, help="Random seed"),
    wandb_project: Optional[str] = typer.Option("grpo-math-experiments", help="Wandb project name"),
) -> None:
    """GRPO training script"""
    
    train_grpo(
        model_name=model_name,
        dataset_path=dataset_path,
        val_dataset_path=val_dataset_path,
        output_dir=output_dir,
        n_grpo_steps=n_grpo_steps,
        learning_rate=learning_rate,
        advantage_eps=advantage_eps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        cliprange=cliprange,
        gradient_clip_value=gradient_clip_value,
        validation_frequency=validation_frequency,
        validation_size=validation_size,
        log_frequency=log_frequency,
        seed=seed,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    typer.run(main) 