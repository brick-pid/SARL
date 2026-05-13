from __future__ import annotations

import logging
from collections import defaultdict
from slime.utils.types import Sample
logger = logging.getLogger(__name__)


def mask_rounds(args, sample_groups) -> None:
    threshold = float(getattr(args, "custom_config", {}).get("mask_reward_threshold", 0.3))

    masked_bucket_count = 0
    masked_sample_count = 0

    for prompt_group in sample_groups:
        buckets: dict[tuple[int, str], list[Sample]] = defaultdict(list)

        for episode_samples in prompt_group:
            for sample in episode_samples:
                round_id = sample.metadata.get("round_id")
                role = sample.metadata.get("role")

                if round_id is None or role is None:
                    continue
                if not _should_consider_bucket(int(round_id), str(role)):
                    continue

                buckets[(int(round_id), str(role))].append(sample)

        for bucket_key, grouped_bucket in buckets.items():
            rewards = [float(sample.get_reward_value(args)) for sample in grouped_bucket]
            has_success = any(reward == 1.0 for reward in rewards)
            mean_reward = sum(rewards) / len(rewards)

            if has_success and mean_reward >= threshold:
                continue

            for sample in grouped_bucket:
                sample.remove_sample = True
            masked_bucket_count += 1
            masked_sample_count += len(grouped_bucket)

            episode_indices = sorted(
                {int(sample.index) for sample in grouped_bucket if sample.index is not None}
            )
            logger.debug(
                "Masked icrl prompt-group bucket %s with mean_reward=%.4f, has_success=%s, size=%d, episode_indices=%s",
                bucket_key,
                mean_reward,
                has_success,
                len(grouped_bucket),
                episode_indices,
            )

    logger.info(
        "icrl sample filter masked %d buckets and %d samples with threshold %.3f",
        masked_bucket_count,
        masked_sample_count,
        threshold,
    )


def _should_consider_bucket(round_id: int, role: str) -> bool:
    if role == "critic":
        return True
    if role == "executor" and round_id >= 2:
        return True
    return False
