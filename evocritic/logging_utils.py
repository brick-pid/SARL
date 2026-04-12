from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .rewards import is_success_reward


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
    _save_rollout_trajectories(rollout_id, args, samples, split="train")

    role_counts = defaultdict(int)
    role_success = defaultdict(int)
    for sample in samples:
        role = sample.metadata.get("role", "unknown")
        role_counts[role] += 1
        reward = sample.reward if isinstance(sample.reward, (int, float)) else 0.0
        if is_success_reward(reward):
            role_success[role] += 1

    for role, count in role_counts.items():
        rollout_extra_metrics[f"rollout/{role}/reward_mean_proxy"] = role_success[role] / max(count, 1)
    rollout_extra_metrics["rollout/episode/success_rate"] = _compute_episode_success_rate(samples)
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    assert extra_metrics is not None
    for dataset_name, dataset_data in data.items():
        samples = dataset_data["samples"]
        _save_rollout_trajectories(rollout_id, args, samples, split="eval")

        role_counts = defaultdict(int)
        role_success = defaultdict(int)
        for sample in samples:
            role = sample.metadata.get("role", "unknown")
            role_counts[role] += 1
            reward = sample.reward if isinstance(sample.reward, (int, float)) else 0.0
            if is_success_reward(reward):
                role_success[role] += 1

        for role, count in role_counts.items():
            extra_metrics[f"eval/{dataset_name}/{role}/reward_mean_proxy"] = role_success[role] / max(count, 1)
        extra_metrics[f"eval/{dataset_name}/episode/success_rate"] = _compute_episode_success_rate(samples)
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

    success_count = sum(is_success_reward(sample.reward) for sample in final_executor_by_episode.values())
    return success_count / len(final_executor_by_episode)


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
            if "task_desc" in sample.metadata:
                f.write(f"task_desc: {sample.metadata['task_desc']}\n")
            trajectory = str(getattr(sample, "trajectory", sample.response))
            f.write(trajectory)
            if not trajectory.endswith("\n"):
                f.write("\n")
            f.write("\n")
            prev_episode_id = episode_id
