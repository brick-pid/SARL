from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .rewards import is_success_reward


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
    _save_rollout_trajectories(rollout_id, args, samples, split="train")
    _log_success_rates(rollout_extra_metrics, "rollout", samples)
    _log_reward_metrics(rollout_extra_metrics, "rollout", samples)
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    assert extra_metrics is not None
    for _, dataset_data in data.items():
        samples = dataset_data["samples"]
        _save_rollout_trajectories(rollout_id, args, samples, split="eval")
        _log_success_rates(extra_metrics, "eval", samples)
        _log_reward_metrics(extra_metrics, "eval", samples)
    return False


def _compute_episode_success_rate(samples) -> float:
    final_executor_by_episode = {}
    for sample in samples:
        if sample.metadata["role"] != "executor":
            continue
        episode_id = sample.index
        round_id = sample.metadata["round_id"]
        prev_sample = final_executor_by_episode.get(episode_id)
        if prev_sample is None or round_id > prev_sample.metadata["round_id"]:
            final_executor_by_episode[episode_id] = sample

    if not final_executor_by_episode:
        return 0.0
    success_count = sum(is_success_reward(sample.reward) for sample in final_executor_by_episode.values())
    return success_count / len(final_executor_by_episode)


def _log_success_rates(metrics: dict, prefix: str, samples) -> None:
    for data_source, data_source_samples in _group_samples_by_metadata(samples, "data_source").items():
        data_source_prefix = f"{prefix}/{data_source}/success_rate"
        metrics[data_source_prefix] = _compute_episode_success_rate(data_source_samples)
        _log_round_success_rates(metrics, data_source_prefix, data_source_samples)

        for subset, subset_samples in _group_samples_by_metadata(data_source_samples, "subset").items():
            subset_prefix = f"{prefix}/{data_source}/{subset}/success_rate"
            metrics[subset_prefix] = _compute_episode_success_rate(subset_samples)
            _log_round_success_rates(metrics, subset_prefix, subset_samples)


def _log_round_success_rates(metrics: dict, prefix: str, samples) -> None:
    max_round_id = 0
    for sample in samples:
        if sample.metadata.get("role") != "executor":
            continue
        round_successes = sample.metadata.get("round_successes")
        if isinstance(round_successes, list) and round_successes:
            max_round_id = max(max_round_id, len(round_successes))
            continue
        max_round_id = max(max_round_id, int(sample.metadata["round_id"]))

    for round_id in range(1, max_round_id + 1):
        metrics[f"{prefix}/round_{round_id}"] = _compute_episode_success_rate_until_round(samples, round_id)


def _compute_episode_success_rate_until_round(samples, max_round_id: int) -> float:
    final_executor_by_episode = {}
    episode_round_successes = {}
    for sample in samples:
        if sample.metadata.get("role") != "executor":
            continue

        episode_id = sample.index
        round_successes = sample.metadata.get("round_successes")
        if isinstance(round_successes, list) and round_successes:
            episode_round_successes[episode_id] = [bool(item) for item in round_successes]
            continue

        round_id = sample.metadata["round_id"]
        if round_id > max_round_id:
            continue

        prev_sample = final_executor_by_episode.get(episode_id)
        if prev_sample is None or round_id > prev_sample.metadata["round_id"]:
            final_executor_by_episode[episode_id] = sample

    success_count = 0
    episode_count = 0

    for round_successes in episode_round_successes.values():
        episode_count += 1
        success_count += int(any(round_successes[:max_round_id]))

    for sample in final_executor_by_episode.values():
        episode_count += 1
        success_count += int(is_success_reward(sample.reward))

    if episode_count == 0:
        return 0.0
    return success_count / episode_count


def _log_reward_metrics(metrics: dict, prefix: str, samples) -> None:
    role_counts = defaultdict(int)
    role_reward_sums = defaultdict(float)
    subset_role_counts = defaultdict(int)
    subset_role_reward_sums = defaultdict(float)

    for sample in samples:
        data_source = sample.metadata.get("data_source")
        if data_source is None:
            raise ValueError("sample.metadata['data_source'] must not be None for icrl logging")
        role = sample.metadata.get("role", "unknown")
        role_counts[(data_source, role)] += 1
        subset = sample.metadata.get("subset")
        if subset:
            subset_role_counts[(data_source, subset, role)] += 1

        reward = sample.reward if isinstance(sample.reward, (int, float)) else 0.0
        role_reward_sums[(data_source, role)] += reward
        if subset:
            subset_role_reward_sums[(data_source, subset, role)] += reward

    for (data_source, role), count in role_counts.items():
        metrics[f"{prefix}/{data_source}/{role}/reward"] = role_reward_sums[(data_source, role)] / max(count, 1)
    for (data_source, subset, role), count in subset_role_counts.items():
        metrics[f"{prefix}/{data_source}/{subset}/{role}/reward"] = (
            subset_role_reward_sums[(data_source, subset, role)] / max(count, 1)
        )


def _group_samples_by_metadata(samples, key: str) -> dict[str, list]:
    grouped = defaultdict(list)
    for sample in samples:
        value = sample.metadata.get(key)
        if value:
            grouped[str(value)].append(sample)
    return dict(grouped)


def _save_rollout_trajectories(rollout_id, args, samples, *, split: str) -> None:
    basedir = Path(args.custom_config["exp_dir"])
    dirname = "rollouts_train" if split == "train" else "rollouts_eval"
    prefix = "train" if split == "train" else "eval"
    filename = f"{prefix}_{rollout_id}.txt"
    output_path = basedir / dirname / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        prev_episode_id = None
        for sample in samples:
            episode_id = sample.index
            if prev_episode_id is not None:
                separator = "--------" if episode_id == prev_episode_id else "========"
                f.write(f"{separator}\n")
            role = sample.metadata.get("role", "unknown")
            f.write(f"episode_id: {episode_id}\n")
            f.write(f"round_id: {sample.metadata.get('round_id')}\n")
            f.write(f"role: {role}\n")
            f.write(f"reward: {sample.reward}\n")
            if "subset" in sample.metadata:
                f.write(f"subset: {sample.metadata['subset']}\n")
            if "task_desc" in sample.metadata:
                f.write(f"task_desc: {sample.metadata['task_desc']}\n")
            trajectory = str(getattr(sample, "trajectory", sample.response))
            f.write(trajectory)
            if not trajectory.endswith("\n"):
                f.write("\n")
            f.write("\n")
            prev_episode_id = episode_id
