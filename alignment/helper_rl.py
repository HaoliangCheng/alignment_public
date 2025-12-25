import torch
from typing import Callable, List, Dict, Tuple, Any, Literal


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages: shape (rollout_batch_size,). Group-normalized rewards for each rollout
                response.
            raw_rewards: shape (rollout_batch_size,). Unnormalized rewards for each rollout
                response.
            metadata: dict with statistics to log (e.g. mean, std, max/min of rewards).
    """
    rollout_batch_size = len(rollout_responses)
    assert len(repeated_ground_truths) == rollout_batch_size
    assert rollout_batch_size % group_size == 0
    
    n_groups = rollout_batch_size // group_size
    
    # Calculate raw rewards for all responses
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards_list.append(reward_dict["reward"])
        format_rewards_list.append(reward_dict["format_reward"])
        answer_rewards_list.append(reward_dict["answer_reward"])
    
    # Convert to tensors
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    format_rewards = torch.tensor(format_rewards_list, dtype=torch.float32)
    answer_rewards = torch.tensor(answer_rewards_list, dtype=torch.float32)
    
    # Reshape to group structure: (n_groups, group_size)
    raw_rewards_grouped = raw_rewards.view(n_groups, group_size)
    
    # Calculate group statistics
    group_means = raw_rewards_grouped.mean(dim=1, keepdim=True)  # (n_groups, 1)
    
    if normalize_by_std:
        # Calculate group standard deviations
        group_stds = raw_rewards_grouped.std(dim=1, keepdim=True)  # (n_groups, 1)
        # Add epsilon to avoid division by zero
        group_stds = torch.clamp(group_stds, min=advantage_eps)
        # Normalize by both mean and std
        advantages_grouped = (raw_rewards_grouped - group_means) / group_stds
    else:
        advantages_grouped = raw_rewards_grouped - group_means

    # Flatten back to original shape
    advantages = advantages_grouped.view(-1)
    
    # Calculate metadata statistics
    metadata = {
        "raw_reward_mean": raw_rewards.mean().item(),
        "raw_reward_std": raw_rewards.std().item(),
        "raw_reward_min": raw_rewards.min().item(),
        "raw_reward_max": raw_rewards.max().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "advantage_min": advantages.min().item(),
        "advantage_max": advantages.max().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
        "group_reward_means": group_means.squeeze().tolist(),
        "n_groups": n_groups,
        "group_size": group_size,
    }
    
    if normalize_by_std:
        group_stds_values = raw_rewards_grouped.std(dim=1, unbiased=False)
        metadata["group_reward_stds"] = group_stds_values.tolist()
        metadata["group_reward_std_mean"] = group_stds_values.mean().item()
    
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
            each token.
    
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    batch_size, sequence_length = policy_log_probs.shape
    
    # Ensure rewards/advantages have the correct shape
    if raw_rewards_or_advantages.dim() == 1:
        # If shape is (batch_size,), reshape to (batch_size, 1)
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)
    
    assert raw_rewards_or_advantages.shape == (batch_size, 1), \
        f"Expected rewards shape ({batch_size}, 1), got {raw_rewards_or_advantages.shape}"
    
    # Broadcast rewards/advantages across sequence length dimension
    # Shape: (batch_size, 1) -> (batch_size, sequence_length)
    broadcasted_rewards = raw_rewards_or_advantages.expand(batch_size, sequence_length)
    
    # Compute policy gradient loss: -reward * log_prob
    # This follows the REINFORCE algorithm where we want to increase the probability of actions that led to higher rewards
    pg_loss = -broadcasted_rewards * policy_log_probs
    
    return pg_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the per-token GRPO-Clip loss.
    
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss: torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
                loss.
            metadata: dict containing clipping statistics and other useful metrics.
    """
    batch_size, sequence_length = policy_log_probs.shape
    
    # Ensure advantages have the correct shape
    if advantages.dim() == 1:
        # If shape is (batch_size,), reshape to (batch_size, 1)
        advantages = advantages.unsqueeze(1)
    
    assert advantages.shape == (batch_size, 1), \
        f"Expected advantages shape ({batch_size}, 1), got {advantages.shape}"
    assert old_log_probs.shape == (batch_size, sequence_length), \
        f"Expected old_log_probs shape ({batch_size}, {sequence_length}), got {old_log_probs.shape}"
    
    # Broadcast advantages across sequence length dimension
    # Shape: (batch_size, 1) -> (batch_size, sequence_length)
    broadcasted_advantages = advantages.expand(batch_size, sequence_length)
    
    # Compute probability ratio: ratio = π_θ(a|s) / π_θ_old(a|s)
    # In log space: log(ratio) = log_prob_new - log_prob_old
    # So: ratio = exp(log_prob_new - log_prob_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # Compute unclipped policy gradient loss: ratio * advantage
    unclipped_loss = ratio * broadcasted_advantages
    
    # Compute clipped ratio within [1-cliprange, 1+cliprange]
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    
    # Compute clipped policy gradient loss: clipped_ratio * advantage
    clipped_loss = clipped_ratio * broadcasted_advantages
    
    # Take the minimum (most conservative) of clipped and unclipped losses
    # This ensures we don't take too large steps that could destabilize training
    final_loss = -torch.min(unclipped_loss, clipped_loss)
    
    # Compute clipping statistics for monitoring
    # A token is considered "clipped" if the clipped loss was chosen over unclipped loss
    is_clipped = (clipped_loss < unclipped_loss)
    
    # Additional useful statistics
    metadata = {
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "ratio_min": ratio.min(),
        "ratio_max": ratio.max(),
        "log_ratio_mean": log_ratio.mean(),
        "log_ratio_std": log_ratio.std(),
        "clip_fraction": is_clipped.float().mean(),  # Fraction of tokens that were clipped
        "is_clipped": is_clipped,  # Per-token clipping indicator
        "unclipped_loss_mean": unclipped_loss.mean(),
        "clipped_loss_mean": clipped_loss.mean(),
        "final_loss_mean": final_loss.mean(),
        "cliprange": torch.tensor(cliprange),
    }
    
    return final_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type: One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar ε used for clipping.
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss: (batch_size, sequence_length), per-token loss.
            metadata: dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    # Validate loss_type
    valid_loss_types = ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    assert loss_type in valid_loss_types, \
        f"loss_type must be one of {valid_loss_types}, got {loss_type}"
    
    # Validate policy_log_probs
    assert policy_log_probs.dim() == 2, \
        f"policy_log_probs must be 2D (batch_size, sequence_length), got shape {policy_log_probs.shape}"
    
    batch_size, sequence_length = policy_log_probs.shape
    
    if loss_type == "no_baseline":
        # Validate required arguments
        assert raw_rewards is not None, \
            "raw_rewards is required when loss_type == 'no_baseline'"
        
        # Delegate to naive policy gradient loss
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        
        # Create metadata (naive policy gradient doesn't return metadata)
        metadata = {
            "loss_type": loss_type,
            "loss_mean": loss.mean(),
            "loss_std": loss.std(),
        }
        
    elif loss_type == "reinforce_with_baseline":
        # Validate required arguments
        assert advantages is not None, \
            "advantages is required when loss_type == 'reinforce_with_baseline'"
        
        # Delegate to naive policy gradient loss with advantages
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        
        # Create metadata (naive policy gradient doesn't return metadata)
        metadata = {
            "loss_type": loss_type,
            "loss_mean": loss.mean(),
            "loss_std": loss.std(),
        }
        
    elif loss_type == "grpo_clip":
        # Validate required arguments
        assert advantages is not None, \
            "advantages is required when loss_type == 'grpo_clip'"
        assert old_log_probs is not None, \
            "old_log_probs is required when loss_type == 'grpo_clip'"
        assert cliprange is not None, \
            "cliprange is required when loss_type == 'grpo_clip'"
        
        # Validate old_log_probs shape
        assert old_log_probs.shape == (batch_size, sequence_length), \
            f"old_log_probs must have shape ({batch_size}, {sequence_length}), got {old_log_probs.shape}"
        
        # Delegate to GRPO clip loss
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        
        # Add loss_type to metadata
        metadata = {
            "loss_type": loss_type,
            **grpo_metadata,  # Include all GRPO-specific metadata
        }
    
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
            masked elements.
    
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    # Validate inputs
    assert tensor.shape == mask.shape, \
        f"tensor and mask must have the same shape, got {tensor.shape} and {mask.shape}"
    
    # Convert mask to float for numerical operations
    mask_float = mask.float()
    
    # Apply mask to tensor: set masked-out elements to 0
    masked_tensor = tensor * mask_float
    
    if dim is None:
        # Compute mean over all masked elements
        total_sum = masked_tensor.sum()
        total_count = mask_float.sum()
        
        # Handle edge case where all elements are masked out
        # Return NaN if no elements are unmasked (consistent with dim!=None behavior)
        if total_count == 0:
            return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        
        return total_sum / total_count
    else:
        # Compute mean along specified dimension
        # Sum the masked values along the dimension
        masked_sum = masked_tensor.sum(dim=dim)
        
        # Count the number of valid (unmasked) elements along the dimension
        mask_count = mask_float.sum(dim=dim)
        
        # Handle edge case where all elements are masked out along the dimension
        safe_mask_count = torch.where(mask_count == 0, torch.ones_like(mask_count), mask_count)
        
        # Compute the mean
        result = masked_sum / safe_mask_count
        
        # Set mean to NaN where all elements were masked out
        result = torch.where(mask_count == 0, torch.full_like(result, float('nan')), result)
        
        return result

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    # Compute the appropriate loss based on loss_type
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    
    # Use masked_mean to aggregate loss over sequence dimension (per-example loss)
    per_example_loss = masked_mean(
        tensor=loss,
        mask=response_mask,
        dim=1
    )
    
    # Average over batch dimension
    aggregated_loss = per_example_loss.mean()
    # Scale loss for gradient accumulation
    scaled_loss = aggregated_loss / gradient_accumulation_steps
    
    # Compute backward pass
    scaled_loss.backward()
    
    # Add additional metadata for logging
    metadata.update({
        "aggregated_loss": aggregated_loss,
        "scaled_loss": scaled_loss
    })
    
    return scaled_loss, metadata