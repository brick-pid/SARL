from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch
from slime.utils.types import Sample

def is_success_reward(reward: float) -> bool:
    # assert 0 <= reward <= 1
    return reward == 1.0 or reward == 100.0

def post_process_rewards(args, samples):
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    return raw_rewards, _prompt_round_role_sample_norm(args, samples, raw_rewards)


def _compute_group_norm_with_pseudo_positive(
    rewards: torch.Tensor,
    *,
    std_norm: bool,
    pseudo_positive_reward: float,
) -> torch.Tensor:
    pseudo_reward = torch.tensor([pseudo_positive_reward], dtype=rewards.dtype, device=rewards.device)
    augmented_rewards = torch.cat([rewards, pseudo_reward])

    mean = augmented_rewards.mean()
    std = augmented_rewards.std() if augmented_rewards.numel() > 1 else torch.tensor(1.0, dtype=rewards.dtype)

    normed = rewards - mean
    if std_norm:
        if augmented_rewards.numel() <= 1:
            return normed
        normed = normed / (std + 1e-6)
    return normed


def _get_prompt_id(sample: Sample) -> int:
    assert sample.group_index is not None, "sample.group_index must not be None"
    return int(sample.group_index)


def _prompt_round_role_group_key(sample: Sample) -> tuple[int, int, str]:
    round_id = sample.metadata.get("round_id")
    role = sample.metadata.get("role")

    if round_id is None:
        raise ValueError("sample.metadata['round_id'] must not be None for icrl reward processing")
    if role is None:
        raise ValueError("sample.metadata['role'] must not be None for icrl reward processing")

    return _get_prompt_id(sample), int(round_id), str(role)


def _prompt_round_role_sample_norm(args, samples, raw_rewards):
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        grouped[_prompt_round_role_group_key(sample)].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)
        normed = _compute_group_norm_with_pseudo_positive(
            rewards,
            std_norm=std_norm,
            pseudo_positive_reward=1.0,
        )
        for i, v in zip(idxs, normed.tolist(), strict=True):
            processed[i] = v
    return processed
