"""
Supervised Fine-Tuning utilities for CS336 Assignment 5.

This module provides functions for tokenizing prompts and outputs for SFT training.
"""

import torch
import logging
from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response 
    tokens and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: Tokenizer to use for tokenization.
        
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          the tokenized prompt and output strings, with the final token sliced off.
        - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          shifted input ids, i.e., the input ids without the first token.
        - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          a mask on the response tokens in the labels.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(f"Number of prompts ({len(prompt_strs)}) must match number of outputs ({len(output_strs)})")
    
    batch_size = len(prompt_strs)
    if batch_size == 0:
        raise ValueError("Empty batch provided")
    
    # Tokenize prompts and outputs separately
    tokenized_prompts = []
    tokenized_outputs = []
    prompt_and_output_lens = []
    
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately without special tokens
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_tokens = tokenizer.encode(output_str, add_special_tokens=False)
        
        tokenized_prompts.append(prompt_tokens)
        tokenized_outputs.append(output_tokens)
        prompt_and_output_lens.append(len(prompt_tokens) + len(output_tokens))
    
    # Find maximum length and prepare for padding
    max_length = max(prompt_and_output_lens)
    
    # Initialize tensors - use 0 for padding initially, then fix pad tokens later
    input_ids = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    labels = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    
    # Fill tensors for each example
    for i, (prompt_tokens, output_tokens) in enumerate(zip(tokenized_prompts, tokenized_outputs)):
        # Concatenate prompt and output tokens
        full_tokens = prompt_tokens + output_tokens
        prompt_len = len(prompt_tokens)
        total_len = len(full_tokens)
        
        # Slice off the final token for input_ids (standard for next-token prediction)
        sequence_len = min(total_len - 1, max_length - 1)
        
        if sequence_len > 0:
            # Fill input_ids (without the last token)
            input_ids[i, :sequence_len] = torch.tensor(full_tokens[:sequence_len], dtype=torch.long)
            
            # Fill labels (shifted by 1, i.e., without the first token)
            labels[i, :sequence_len] = torch.tensor(full_tokens[1:sequence_len + 1], dtype=torch.long)
            
            # Create response mask: 1 for response tokens, 0 for prompt tokens
            # Response tokens start after the prompt in the labels tensor
            # Since labels is shifted by 1, response starts at (prompt_len - 1)
            response_start = max(0, prompt_len - 1)
            response_end = sequence_len
            
            if response_start < response_end:
                response_mask[i, response_start:response_end] = 1
    
    # Handle padding: set pad positions to the appropriate pad token
    # For positions beyond sequence length, we need to set them to pad token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    
    for i, (prompt_tokens, output_tokens) in enumerate(zip(tokenized_prompts, tokenized_outputs)):
        total_len = len(prompt_tokens) + len(output_tokens)
        sequence_len = min(total_len-1 , max_length-1)
        
        # Set padding positions to pad token
        if sequence_len < max_length - 1:
            input_ids[i, sequence_len+1:] = pad_token_id  
            labels[i, sequence_len:] = pad_token_id  
    
    
    return {
        "input_ids": input_ids,
        "labels": labels, 
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.
                
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    # Compute log probabilities using log_softmax for numerical stability
    # log_softmax(x) = x - logsumexp(x) which is numerically stable
    log_probs = torch.log_softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    
    # Compute probabilities from log probabilities
    probs = torch.exp(log_probs)  # shape: (batch_size, seq_len, vocab_size)
    
    # Compute entropy: H = -∑ p(x) * log(p(x))
    # This is equivalent to: H = -∑ probs * log_probs
    entropy = -torch.sum(probs * log_probs, dim=-1)  # shape: (batch_size, seq_len)
    
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model's next-token distribution.
    
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
               and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
                   response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
                tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling compute_entropy.
        
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt|x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
                        for each position (present only if return_token_entropy=True).
    """

    if model.training:
        # Training mode - keep gradients enabled
        outputs = model(input_ids)
        logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)
    else:
        # Evaluation mode - disable gradients for efficiency
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
    
    # Compute log probabilities using log_softmax for numerical stability
    log_probs_dist = torch.log_softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    
    # Get log probabilities for the actual tokens in labels
    # We need to gather the log probs for the specific tokens
    batch_size, seq_len = labels.shape
    
    # Create a mask for valid (non-ignored) labels
    # Labels with value -100 are typically ignored in loss computation
    valid_mask = (labels != -100)
    
    # Initialize log_probs tensor
    log_probs = torch.full_like(labels, 0.0, dtype=torch.float32)
    
    # Only compute log probs for valid positions
    if valid_mask.any():
        # Get the indices where labels are valid
        valid_labels = labels.clone()
        valid_labels[~valid_mask] = 0  # Set invalid labels to 0 temporarily for gathering
        
        # Gather log probabilities for the actual tokens
        # torch.gather gathers values along the vocab dimension using labels as indices
        gathered_log_probs = torch.gather(
            log_probs_dist, 
            dim=-1, 
            index=valid_labels.unsqueeze(-1)
        ).squeeze(-1)  # shape: (batch_size, sequence_length)
        
        # Only keep log probs for valid positions, set others to 0
        log_probs = gathered_log_probs * valid_mask.float()
    
    # Prepare return dictionary
    result = {"log_probs": log_probs}
    
    # Optionally compute and return token entropy
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
             dimensions.
             
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to
        the sum.
    """
    # Verify tensor and mask have the same shape
    if tensor.shape != mask.shape:
        raise ValueError(f"Tensor and mask must have the same shape. "
                        f"Got tensor: {tensor.shape}, mask: {mask.shape}")
    
    # Apply mask to tensor - only elements where mask == 1 contribute
    # Convert mask to same dtype as tensor for proper multiplication
    mask_float = mask.float()
    masked_tensor = tensor * mask_float
    
    # Sum over the specified dimension(s)
    if dim is None:
        # Sum over all dimensions
        masked_sum = torch.sum(masked_tensor)
    else:
        # Sum over the specified dimension
        masked_sum = torch.sum(masked_tensor, dim=dim)
    
    # Normalize by the constant
    if normalize_constant == 0:
        raise ValueError("normalize_constant cannot be zero")
    
    normalized_result = masked_sum / normalize_constant
    
    return normalized_result

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs: torch.Tensor (batch_size, sequence_length), per-token log-probabilities 
                         from the SFT policy being trained.
        response_mask: torch.Tensor (batch_size, sequence_length), 1 for response tokens, 
                      0 for prompt/padding.
        gradient_accumulation_steps: int Number of microbatches per optimizer step.
        normalize_constant: float The constant by which to divide the sum. It is fine to leave this as 1.0.
        
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. 
              We return this so we can log it.
        metadata: Dict with metadata from the underlying loss call, and any other statistics 
                 you might want to log.
    """
    # Verify input shapes match
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(f"policy_log_probs and response_mask must have the same shape. Got {policy_log_probs.shape} and {response_mask.shape}")
    
    # Compute cross-entropy loss: we want to maximize log probabilities, so minimize negative log probs
    # For SFT, the loss is the negative log likelihood of the target tokens
    negative_log_probs = -policy_log_probs
    
    # Use masked_normalize to compute the loss only over response tokens
    # Sum over all dimensions and normalize by the provided constant
    # Special case: if normalize_constant is 1.0, just return the sum without normalization

    masked_loss = masked_normalize(
        tensor=negative_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None 
    )
    
    # Scale loss for gradient accumulation and batch averaging
    # Normalize by batch size to get average loss per example
    # Then divide by gradient_accumulation_steps for gradient accumulation
    batch_size = policy_log_probs.shape[0]
    scaled_loss = masked_loss / (batch_size * gradient_accumulation_steps)
    
    # Compute backward pass
    scaled_loss.backward()
    
    # Compute metadata for logging
    with torch.no_grad():
        # Count number of response tokens for normalization info
        num_response_tokens = response_mask.sum()
        
        # Compute average loss per response token
        if num_response_tokens > 0:
            avg_loss_per_token = masked_loss / num_response_tokens
        else:
            avg_loss_per_token = torch.tensor(0.0, device=masked_loss.device)
        
        # Compute some useful statistics
        metadata = {
            "num_response_tokens": num_response_tokens,
            "avg_loss_per_token": avg_loss_per_token,
            "unscaled_loss": masked_loss,  # Loss before gradient accumulation scaling
            "policy_log_probs_mean": masked_normalize(
                policy_log_probs, response_mask, 
                normalize_constant=max(num_response_tokens.item(), 1),
                dim=None
            ),
            "policy_log_probs_std": torch.std(policy_log_probs[response_mask.bool()]) if num_response_tokens > 0 else torch.tensor(0.0)
        }
    
    return scaled_loss, metadata


def log_generations(
    prompts: List[str],
    generations: List[str],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[dict]] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    max_examples: int = 5,
    logger: Optional[logging.Logger] = None,
    prefix: str = "GENERATION"
) -> None:
    """
    Log generations from a model for monitoring and debugging purposes.
    Focuses on core metrics: prompts, generations, ground truth, rewards, entropy, and response lengths.
    
    Args:
        prompts: List of input prompts.
        generations: List of model generations corresponding to prompts.
        ground_truths: Optional list of ground truth responses.
        metrics: Optional list of metric dictionaries for each example.
                Expected keys in metrics dict:
                - 'format_reward': reward for correct format
                - 'answer_reward': reward for correct answer  
                - 'total_reward': total reward score
                - 'token_entropy': per-token entropy values or average entropy
                - 'is_correct': boolean indicating if response is correct
        step: Optional training step number.
        epoch: Optional training epoch number.
        max_examples: Maximum number of examples to log (to avoid spam).
        logger: Optional logger instance. If None, uses default logger.
        prefix: Prefix for log messages.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate inputs
    if len(prompts) != len(generations):
        raise ValueError(f"Number of prompts ({len(prompts)}) must match number of generations ({len(generations)})")
    
    if ground_truths is not None and len(ground_truths) != len(prompts):
        raise ValueError(f"Number of ground truths ({len(ground_truths)}) must match number of prompts ({len(prompts)})")
    
    if metrics is not None and len(metrics) != len(prompts):
        raise ValueError(f"Number of metrics ({len(metrics)}) must match number of prompts ({len(prompts)})")
    
    # Prepare header info
    header_parts = [prefix]
    if step is not None:
        header_parts.append(f"Step {step}")
    if epoch is not None:
        header_parts.append(f"Epoch {epoch}")
    
    header = " ".join(header_parts)
    
    # Limit number of examples to avoid overwhelming output
    num_examples = min(len(prompts), max_examples)
    
    logger.info(f"{header} - Showing {num_examples}/{len(prompts)} examples")
    logger.info("=" * 60)
    
    # Log each example with core information only
    for i in range(num_examples):
        logger.info(f"\n--- Example {i+1} ---")
        
        # 1. Input prompt
        logger.info(f"PROMPT: {prompts[i]}")
        
        # 2. Response generated by the model
        logger.info(f"GENERATION: {generations[i]}")
        
        # 3. Ground-truth answer
        if ground_truths is not None:
            logger.info(f"GROUND_TRUTH: {ground_truths[i]}")
        
        # 4. Reward information (format, answer, total)
        if metrics is not None:
            metric_dict = metrics[i]
            
            reward_parts = []
            if 'format_reward' in metric_dict:
                reward_parts.append(f"format: {metric_dict['format_reward']:.3f}")
            if 'answer_reward' in metric_dict:
                reward_parts.append(f"answer: {metric_dict['answer_reward']:.3f}")
            if 'total_reward' in metric_dict:
                reward_parts.append(f"total: {metric_dict['total_reward']:.3f}")
            
            if reward_parts:
                logger.info(f"REWARDS: {', '.join(reward_parts)}")
            
            # 5. Average token entropy of the response
            if 'token_entropy' in metric_dict:
                entropy_val = metric_dict['token_entropy']
                if isinstance(entropy_val, torch.Tensor):
                    if entropy_val.numel() > 1:
                        avg_entropy = entropy_val.mean().item()
                    else:
                        avg_entropy = entropy_val.item()
                else:
                    avg_entropy = entropy_val
                logger.info(f"AVG_TOKEN_ENTROPY: {avg_entropy:.4f}")
    
    logger.info("=" * 60)
    
    # 6. Response length statistics
    if metrics is not None:
        log_core_aggregate_metrics(metrics, generations, logger)


def log_core_aggregate_metrics(
    metrics: List[dict], 
    generations: List[str],
    logger: logging.Logger
) -> None:
    """Log core aggregate statistics focusing on response length analysis."""
    if not metrics:
        return
    
    # Compute response lengths for each generation
    response_lengths = [len(gen.split()) for gen in generations]
    
    logger.info("Aggregate Metrics:")
    
    # Overall average response length
    avg_length = sum(response_lengths) / len(response_lengths)
    logger.info(f"  avg_response_length: {avg_length:.2f} tokens")
    
    # Length analysis by correctness
    correct_lengths = []
    incorrect_lengths = []
    
    for i, metric_dict in enumerate(metrics):
        if 'is_correct' in metric_dict:
            if metric_dict['is_correct']:
                correct_lengths.append(response_lengths[i])
            else:
                incorrect_lengths.append(response_lengths[i])
    
    if correct_lengths:
        avg_correct_length = sum(correct_lengths) / len(correct_lengths)
        logger.info(f"  avg_response_length_correct: {avg_correct_length:.2f} tokens ({len(correct_lengths)} examples)")
    
    if incorrect_lengths:
        avg_incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths)
        logger.info(f"  avg_response_length_incorrect: {avg_incorrect_length:.2f} tokens ({len(incorrect_lengths)} examples)")



