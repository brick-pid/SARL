from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .experience_bank import Critique, Trajectory, load_experience_bank
from .rewards import is_success_reward


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
    _save_rollout_trajectories(rollout_id, args, samples, split="train")
    _write_experience_bank(args, samples)

    _log_group_metrics(rollout_extra_metrics, "rollout", samples)
    for data_source, data_source_samples in _group_samples_by_metadata(samples, "data_source").items():
        rollout_extra_metrics[f"rollout/{data_source}/success_rate"] = _compute_episode_success_rate(data_source_samples)
        _log_round_success_rates(rollout_extra_metrics, f"rollout/{data_source}/success_rate", data_source_samples)
    for subset_name, subset_samples in _group_samples_by_subset(samples).items():
        rollout_extra_metrics[f"rollout/{subset_name}/success_rate"] = _compute_episode_success_rate(subset_samples)
        _log_round_success_rates(rollout_extra_metrics, f"rollout/{subset_name}/success_rate", subset_samples)
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    assert extra_metrics is not None
    for dataset_name, dataset_data in data.items():
        samples = dataset_data["samples"]
        _save_rollout_trajectories(rollout_id, args, samples, split="eval")

        _log_group_metrics(extra_metrics, f"eval/{dataset_name}", samples)
        for data_source, data_source_samples in _group_samples_by_metadata(samples, "data_source").items():
            extra_metrics[f"eval/{dataset_name}/{data_source}/success_rate"] = _compute_episode_success_rate(
                data_source_samples
            )
            _log_round_success_rates(
                extra_metrics,
                f"eval/{dataset_name}/{data_source}/success_rate",
                data_source_samples,
            )
        for subset_name, subset_samples in _group_samples_by_subset(samples).items():
            extra_metrics[f"eval/{dataset_name}/{subset_name}/success_rate"] = _compute_episode_success_rate(
                subset_samples
            )
            _log_round_success_rates(
                extra_metrics,
                f"eval/{dataset_name}/{subset_name}/success_rate",
                subset_samples,
            )
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


def _log_round_success_rates(metrics: dict, prefix: str, samples) -> None:
    max_round_id = 0
    for sample in samples:
        if sample.metadata.get("role") != "executor":
            continue
        round_successes = sample.metadata.get("round_successes")
        if isinstance(round_successes, list) and round_successes:
            round_id = len(round_successes)
        else:
            round_id = int(sample.metadata["round_id"])
        max_round_id = max(max_round_id, round_id)

    for round_id in range(1, max_round_id + 1):
        metrics[f"{prefix}/round_{round_id}"] = _compute_episode_success_rate_until_round(samples, round_id)


def _compute_episode_success_rate_until_round(samples, max_round_id: int) -> float:
    final_executor_by_episode = {}
    episode_round_successes = {}
    for sample in samples:
        if sample.metadata["role"] != "executor":
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


def _log_group_metrics(metrics: dict, prefix: str, samples) -> None:
    role_counts = defaultdict(int)
    role_reward_sums = defaultdict(float)
    subset_role_counts = defaultdict(int)
    subset_role_reward_sums = defaultdict(float)

    for sample in samples:
        role = sample.metadata.get("role", "unknown")
        role_counts[role] += 1
        subset = sample.metadata.get("subset")
        if subset:
            subset_role_counts[(subset, role)] += 1

        reward = sample.reward if isinstance(sample.reward, (int, float)) else 0.0
        role_reward_sums[role] += reward
        if subset:
            subset_role_reward_sums[(subset, role)] += reward

    for role, count in role_counts.items():
        metrics[f"{prefix}/{role}/reward"] = role_reward_sums[role] / max(count, 1)
    for (subset, role), count in subset_role_counts.items():
        metrics[f"{prefix}/{subset}/{role}/reward"] = subset_role_reward_sums[(subset, role)] / max(count, 1)


def _group_samples_by_subset(samples) -> dict[str, list]:
    return _group_samples_by_metadata(samples, "subset")


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


def _write_experience_bank(args, samples) -> None:
    bank = load_experience_bank(args.custom_config)
    changed = False
    for sample in samples:
        role = sample.metadata.get("role")
        task_id = sample.metadata.get("task_id")
        task_desc = sample.metadata.get("task_desc")
        if not task_id or not task_desc:
            continue

        if role == "executor":
            trajectory = _build_success_trajectory(sample)
            if trajectory is not None:
                existing_entry = bank.entries.get(task_id)
                before_turn = existing_entry.trajectory.turn if existing_entry and existing_entry.trajectory else None
                bank.update_success_trajectory(task_id, task_desc, trajectory)
                after = bank.entries.get(task_id)
                after_turn = after.trajectory.turn if after and after.trajectory else None
                changed = changed or after_turn != before_turn
        elif role == "critic":
            critique = _build_valid_critique(sample)
            if critique is not None:
                bank.add_critique(task_id, task_desc, critique)
                changed = True

    if changed:
        bank.save()


def _build_success_trajectory(sample) -> Trajectory | None:
    reward = sample.reward
    turn = sample.metadata.get("turn")
    steps = sample.metadata.get("trajectory_steps")
    task_desc = sample.metadata.get("task_desc")
    if reward != 1.0 or task_desc is None or turn is None or not isinstance(steps, list):
        return None
    return Trajectory(
        task_desc=task_desc,
        turn=int(turn),
        reward=float(reward),
        steps=[
            {
                "action": str(step.get("action", "")),
                "observation": str(step.get("observation", "")),
            }
            for step in steps
        ],
    )


def _build_valid_critique(sample) -> Critique | None:
    task = sample.metadata.get("task_desc")
    text = sample.metadata.get("critique_text")
    reward_prev = sample.metadata.get("reward_prev")
    reward_after = sample.metadata.get("reward_after")
    reward_diff = sample.metadata.get("reward_diff")
    if not task or not text or reward_prev is None or reward_after is None or reward_diff is None:
        return None
    reward_diff = float(reward_diff)
    if reward_diff <= 0.1:
        return None
    return Critique(
        task=str(task),
        text=str(text),
        reward_prev=float(reward_prev),
        reward_after=float(reward_after),
        reward_diff=reward_diff,
    )
