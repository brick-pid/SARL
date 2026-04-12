from __future__ import annotations

from dataclasses import dataclass, field

from slime.utils.types import Sample


@dataclass
class RoundRecord:
    round_id: int
    executor_sample: Sample
    executor_reward: float
    verifier_sample: Sample
    verifier_pred_success: bool
    verifier_reward: float
    critic_sample: Sample | None = None
    critic_reward: float | None = None


@dataclass
class EpisodeRecord:
    task: str
    data_source: str
    rounds: list[RoundRecord] = field(default_factory=list)
