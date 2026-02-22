"""
A simple tool parser

support format:
<action> act_str </action>
<subagent> act_str </subagent>
"""
import re
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ParseResult:
    type: str # in ("action", "subagent")
    content: str # action/subagent input/conclusion

def tool_parser(response: str) -> ParseResult | None:
    pattern = r"<(action|subagent|conclusion)>(.*?)</\1>"
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if not matches:
        return None
    m = matches[-1]
    return ParseResult(type=m.group(1), content=m.group(2).strip())

def _is_success(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value in {1, 100}
    return 0

def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    if rollout_extra_metrics is None:
        rollout_extra_metrics = {}

    success_count = 0
    for sample in samples:
        success_count += _is_success(getattr(sample, "reward", None))

    rollout_extra_metrics["rollout/success_rate"] = success_count / len(samples)
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    if rollout_extra_metrics is None:
        rollout_extra_metrics = {}

    for dataset_name, dataset_data in data.items():
        rewards = dataset_data.get("rewards", [])
        success_count = 0
        for r in rewards:
            success_count += _is_success(r)
        extra_metrics[f"eval/{dataset_name}/success_rate"] = success_count / len(rewards)
    return False
