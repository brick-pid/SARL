from __future__ import annotations

from collections import defaultdict
import torch
from slime.utils.types import Sample


def post_process_rewards(args, samples):
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    norm = getattr(args, "reward_norm", "sample_norm")
    if norm == "sample_norm":
        return raw_rewards, _role_sample_norm(args, samples, raw_rewards)
    if norm == "episode_norm":
        return raw_rewards, _role_episode_norm(args, samples, raw_rewards)
    raise ValueError(f"Unknown reward_norm: {norm}")


def _compute_group_norm(rewards, augment, std_norm):
    if augment not in [None, "none", "pseudo_pos"]:
        raise ValueError(f"Unknown reward_augment: {augment}")
    all_zero = (rewards == 0).all().item()
    if augment == "pseudo_pos" and all_zero:
        if rewards.numel() == 1:
            mean = torch.tensor(0.0, dtype=torch.float)
            std = torch.tensor(1.0, dtype=torch.float)
        else:
            stats_rewards = torch.cat([rewards, torch.tensor([1.0], dtype=torch.float)], dim=0)
            mean = stats_rewards.mean()
            std = stats_rewards.std()
    else:
        mean = rewards.mean()
        std = rewards.std() if rewards.numel() > 1 else torch.tensor(1.0, dtype=torch.float)

    normed = rewards - mean
    if std_norm:
        if not (augment == "pseudo_pos" and all_zero):
            assert rewards.numel() > 1, "std normalization requires at least 2 elements in each group"
        normed = normed / (std + 1e-6)
    return normed


def _role_group_key(sample: Sample) -> tuple[int, str]:
    assert sample.group_index is not None, "sample.group_index must not be None"
    role = sample.metadata.get("role")
    if role is None:
        raise ValueError("sample.metadata['role'] must not be None for EvoCritic reward processing")
    return sample.group_index, role


def _role_sample_norm(args, samples, raw_rewards):
    augment = getattr(args, "reward_augment", None)
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        grouped[_role_group_key(sample)].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)
        normed = _compute_group_norm(rewards, augment, std_norm)
        for i, v in zip(idxs, normed.tolist(), strict=True):
            processed[i] = v
    return processed


def _role_episode_norm(args, samples, raw_rewards):
    augment = getattr(args, "reward_augment", None)
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        grouped[_role_group_key(sample)].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        episode_to_idxs = defaultdict(list)
        for i in idxs:
            assert samples[i].index is not None, "sample.index must not be None"
            episode_to_idxs[samples[i].index].append(i)

        episode_ids = list(episode_to_idxs.keys())
        rewards = torch.tensor([max(raw_rewards[j] for j in episode_to_idxs[eid]) for eid in episode_ids], dtype=torch.float)
        normed = _compute_group_norm(rewards, augment, std_norm)

        for eid, v in zip(episode_ids, normed.tolist(), strict=True):
            for i in episode_to_idxs[eid]:
                processed[i] = v
    return processed

