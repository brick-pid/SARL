from collections import defaultdict
import torch


def post_process_rewards_v1(args, samples):
    """
    v1: sample-level normalization within each group_index.
    This keeps trajectory-length weight bias.
    """
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # group norm
        grouped = defaultdict(list)
        for i, sample in enumerate(samples):
            assert sample.group_index is not None, "sample.group_index must not be None"
            grouped[sample.group_index].append(i)

        processed = [0.0] * len(samples)
        for idxs in grouped.values():
            rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)
            mean = rewards.mean()
            rewards = rewards - mean

            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                assert rewards.numel() > 1, "std normalization requires at least 2 samples in each group"
                std = rewards.std()
                rewards = rewards / (std + 1e-6)

            for i, v in zip(idxs, rewards.tolist(), strict=True):
                processed[i] = v

        return raw_rewards, processed

    return raw_rewards, raw_rewards


def post_process_rewards_v2(args, samples):
    """
    v2: trajectory-level normalization within each group_index, then broadcast
    to all samples in the trajectory. This removes subagent-count weight bias.
    We assume samples from the same trajectory have the same reward value.
    """
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # group norm (trajectory-level, then broadcast)
        grouped = defaultdict(list)
        for i, sample in enumerate(samples):
            assert sample.group_index is not None, "sample.group_index must not be None"
            grouped[sample.group_index].append(i)

        processed = [0.0] * len(samples)
        for idxs in grouped.values():
            traj_to_idxs = defaultdict(list)
            for i in idxs:
                assert samples[i].index is not None, "sample.index must not be None"
                traj_id = samples[i].index
                traj_to_idxs[traj_id].append(i)

            traj_ids = list(traj_to_idxs.keys())
            # use the reward of the first sample in the trajectory as the trajectory reward
            for t in traj_ids:
                idxs_in_traj = traj_to_idxs[t]
                expected = raw_rewards[idxs_in_traj[0]]
                assert all(raw_rewards[j] == expected for j in idxs_in_traj), (
                    f"Trajectory {t} has inconsistent rewards: "
                    f"{[raw_rewards[j] for j in idxs_in_traj]}"
                )
            rewards = torch.tensor([raw_rewards[traj_to_idxs[t][0]] for t in traj_ids], dtype=torch.float)
            mean = rewards.mean()
            rewards = rewards - mean

            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                assert rewards.numel() > 1, "std normalization requires at least 2 trajectories in each group"
                std = rewards.std()
                rewards = rewards / (std + 1e-6)

            for t, v in zip(traj_ids, rewards.tolist(), strict=True):
                for i in traj_to_idxs[t]:
                    processed[i] = v

        return raw_rewards, processed

    return raw_rewards, raw_rewards


def post_process_rewards_v3(args, samples):
    """
    v3: based on v1 sample-level normalization.
    For each group_index, append one max_reward sample when computing mean/std.
    """
    assert not (samples and isinstance(samples[0], list)), "samples should be flattened before reward post-process"

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        # group norm
        grouped = defaultdict(list)
        for i, sample in enumerate(samples):
            assert sample.group_index is not None, "sample.group_index must not be None"
            grouped[sample.group_index].append(i)

        processed = [0.0] * len(samples)

        for idxs in grouped.values():
            rewards = torch.tensor([raw_rewards[i] for i in idxs], dtype=torch.float)

            if rewards.numel() == 1:
                mean = torch.tensor(0.0, dtype=torch.float)
                std = torch.tensor(1.0, dtype=torch.float)
            else:
                max_rewards = torch.tensor([1.0], dtype=torch.float)
                stats_rewards = torch.cat([rewards, max_rewards], dim=0)
                mean = stats_rewards.mean()
                std = stats_rewards.std()

            rewards = rewards - mean
            if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
                rewards = rewards / (std + 1e-6)

            for i, v in zip(idxs, rewards.tolist(), strict=True):
                processed[i] = v

        return raw_rewards, processed

    return raw_rewards, raw_rewards
