from collections import defaultdict
import torch


def post_process_rewards(args, samples):
    """Entry point: dispatch to norm strategy based on args.reward_norm."""
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    norm = getattr(args, "reward_norm", "sample_norm")
    if norm == "sample_norm":
        return raw_rewards, _sample_norm(args, samples, raw_rewards)
    elif norm == "episode_norm":
        return raw_rewards, _episode_norm(args, samples, raw_rewards)
    else:
        raise ValueError(f"Unknown reward_norm: {norm}")


def _compute_group_norm(rewards, augment, std_norm):
    """Shared normalization logic for a group of rewards.

    Args:
        rewards: 1-D tensor of rewards to normalize.
        augment: "none" or "pseudo_pos".
        std_norm: whether to divide by std.

    Returns:
        Normalized rewards tensor (same shape as input).
    """
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


def _sample_norm(args, samples, raw_rewards):
    """Sample-level normalization within each group_index (original v1 / v3)."""
    augment = getattr(args, "reward_augment", None)
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        assert sample.group_index is not None, "sample.group_index must not be None"
        grouped[sample.group_index].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)
        normed = _compute_group_norm(rewards, augment, std_norm)
        for i, v in zip(idxs, normed.tolist(), strict=True):
            processed[i] = v

    return processed


def _episode_norm(args, samples, raw_rewards):
    """Trajectory-level normalization within each group_index, then broadcast (original v2)."""
    augment = getattr(args, "reward_augment", None)
    std_norm = args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization

    grouped = defaultdict(list)
    for i, sample in enumerate(samples):
        assert sample.group_index is not None, "sample.group_index must not be None"
        grouped[sample.group_index].append(i)

    processed = [0.0] * len(samples)
    for idxs in grouped.values():
        traj_to_idxs = defaultdict(list)
        for i in idxs:
            assert samples[i].index is not None, "sample.index must not be None"
            traj_to_idxs[samples[i].index].append(i)

        traj_ids = list(traj_to_idxs.keys())
        for t in traj_ids:
            idxs_in_traj = traj_to_idxs[t]
            expected = raw_rewards[idxs_in_traj[0]]
            assert all(raw_rewards[j] == expected for j in idxs_in_traj), (
                f"Trajectory {t} has inconsistent rewards: "
                f"{[raw_rewards[j] for j in idxs_in_traj]}"
            )

        rewards = torch.tensor([raw_rewards[traj_to_idxs[t][0]] for t in traj_ids], dtype=torch.float)
        normed = _compute_group_norm(rewards, augment, std_norm)

        for t, v in zip(traj_ids, normed.tolist(), strict=True):
            for i in traj_to_idxs[t]:
                processed[i] = v

    return processed
