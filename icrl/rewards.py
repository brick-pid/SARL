from __future__ import annotations

from collections import defaultdict
import torch
from slime.utils.types import Sample


def _get_reward_norm(args) -> str:
    return args.custom_config['reward_norm']


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

    norm = _get_reward_norm(args)
    if norm == "role_norm":
        return raw_rewards, _role_sample_norm(args, samples, raw_rewards)
    raise ValueError(f"Unknown reward_norm: {norm}")


def _compute_group_norm(rewards, std_norm):
    mean = rewards.mean()
    std = rewards.std() if rewards.numel() > 1 else torch.tensor(1.0, dtype=torch.float)

    normed = rewards - mean
    if std_norm:
        if rewards.numel() <= 1:
            return normed
        normed = normed / (std + 1e-6)
    return normed


def _role_group_key(sample: Sample) -> tuple[int, str]:
    assert sample.group_index is not None, "sample.group_index must not be None"
    role = sample.metadata.get("role")
    if role is None:
        raise ValueError("sample.metadata['role'] must not be None for icrl reward processing")
    return sample.group_index, role


def _role_sample_norm(args, samples, raw_rewards):
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        grouped[_role_group_key(sample)].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)
        normed = _compute_group_norm(rewards, std_norm)
        for i, v in zip(idxs, normed.tolist(), strict=True):
            processed[i] = v
    return processed
