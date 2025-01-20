from typing import Optional, Tuple, Union

from numpy import float32
import torch
import torch.nn.functional as F

def compute_reward_new(
    r: list[list[float]],
    eostep_indices: list[list[int]],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """
    Modified reward computation function that handles variable-length rewards and custom indices.
    
    Args:
        r: List of lists containing reward values for each sequence
        kl_coef: KL divergence coefficient
        kl: KL divergence values as Tensor or list of Tensors
        action_mask: Optional mask for actions
        num_actions: Optional number of actions
        reward_clip_range: Optional range for reward clipping
        eostep_indices: List of lists containing indices where rewards should be assigned
    
    Returns:
        Tensor or list of Tensors containing computed rewards
    """
    if kl_coef <= 0.0:
        kl_coef = 0.0

    # Convert rewards to tensor and apply clipping if needed
    max_seq_len = action_mask.size(1) if action_mask is not None else max(len(indices) for indices in eostep_indices)
    batch_size = len(r)
    
    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # Initialize reward tensor
        last_reward = torch.zeros_like(kl)
        
        # Distribute rewards to specified indices
        for i in range(batch_size):
            assert len(r[i]) == len(eostep_indices[i])
            curr_r = torch.tensor(r[i], device=kl.device, dtype=torch.float32)
            if reward_clip_range:
                curr_r = curr_r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])
            
            # Convert indices to tensor
            curr_indices = torch.tensor(eostep_indices[i], device=kl.device, dtype=torch.long)
            
            # Create index tensor for scatter operation
            batch_idx = torch.full_like(curr_indices, i)
            indices_2d = torch.stack([batch_idx, curr_indices], dim=0)
            
            # Distribute rewards
            last_reward.index_put_(
                (indices_2d[0], indices_2d[1]),
                curr_r,
                accumulate=True
            )
            
        reward = last_reward + kl_reward
    else:
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            curr_r = torch.tensor(r[i], device=kl_seg.device)
            if reward_clip_range:
                curr_r = curr_r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])
            
            # Add rewards at specified indices
            for idx, r_val in zip(eostep_indices[i], curr_r):
                kl_reward[idx] += r_val
                
            reward.append(kl_reward)
    
    return reward


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs.float() - log_probs_base.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values
