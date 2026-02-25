"""
A simple tool parser

support format:
<action> act_str </action>
<subagent> act_str </subagent>
"""
import re
from dataclasses import dataclass
from collections import defaultdict


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
    assert rollout_extra_metrics is not None
    success_count = 0
    subset_count = defaultdict(int)
    subset_success_count = defaultdict(int)
    for sample in samples:
        success = _is_success(sample.reward)
        success_count += success
        # subset
        subset = sample.metadata['subset']
        subset_count[subset] += 1
        subset_success_count[subset] += success

    rollout_extra_metrics["rollout/success_rate"] = success_count / len(samples)
    for subset in subset_count:
        rollout_extra_metrics[f"rollout/{subset}/success_rate"] = subset_success_count[subset] / subset_count[subset]
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    assert extra_metrics is not None
    for dataset_name, dataset_data in data.items():
        samples = dataset_data['samples']
        success_count = 0
        subset_count = defaultdict(int)
        subset_success_count = defaultdict(int)
        for sample in samples:
            success = _is_success(sample.reward)
            success_count += success
            # subset
            subset = sample.metadata['subset']
            subset_count[subset] += 1
            subset_success_count[subset] += success

        extra_metrics[f"eval/{dataset_name}/success_rate"] = success_count / len(samples)    
        for subset in subset_count:
            extra_metrics[f"eval/{dataset_name}/{subset}/success_rate"] = subset_success_count[subset] / subset_count[subset]
    return False
