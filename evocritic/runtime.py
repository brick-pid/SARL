from __future__ import annotations

from copy import deepcopy
from typing import Any

from slime.utils.http_utils import post
from slime.utils.types import Sample


def clone_role_sample(
    base_sample: Sample,
    *,
    role: str,
    round_id: int,
    prompt: str | list[dict[str, str]],
) -> Sample:
    sample = deepcopy(base_sample)
    sample.prompt = prompt
    sample.tokens = []
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.response = ""
    sample.response_length = 0
    sample.reward = None
    sample.outcome_reward = None
    sample.subagent_bonus = None
    sample.metadata = dict(base_sample.metadata)
    sample.metadata["role"] = role
    sample.metadata["round_id"] = round_id
    return sample


def init_sample_state(sample: Sample, prompt_ids: list[int]) -> list[int]:
    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    return []


def append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    *,
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def append_observation_turn(
    *,
    sample: Sample,
    response_tokens: list[int],
    tokenizer,
    turn_pre: list[int],
    turn_post: list[int],
    obs: str,
) -> int:
    obs_ids = tokenizer.encode(obs, add_special_tokens=False)
    turn_ids = turn_pre + obs_ids + turn_post
    append_to_sample(sample, response_tokens, turn_ids, [0.0] * len(turn_ids), loss_mask_val=0)
    return len(turn_ids)


def build_chat_turn_markers(tokenizer) -> tuple[list[int], list[int]]:
    turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    return turn_pre, turn_post


async def generate_one_turn(
    *,
    input_ids: list[int],
    url: str,
    sampling_params: dict[str, Any],
    budget: int,
) -> tuple[str, list[int], list[float]]:
    cur_params = sampling_params.copy()
    cur_params["max_new_tokens"] = budget
    payload = {"input_ids": input_ids, "sampling_params": cur_params, "return_logprob": True}
    output = await post(url, payload)
    raw_logprobs = output["meta_info"]["output_token_logprobs"]
    new_token_ids = [item[1] for item in raw_logprobs]
    new_log_probs = [item[0] for item in raw_logprobs]
    return output["text"], new_token_ids, new_log_probs


def should_stop_on_repeat(action_list: list[str], max_repeat: int) -> bool:
    if len(action_list) < max_repeat:
        return False
    recent_actions = action_list[-max_repeat:]
    return len(set(recent_actions)) == 1


def finalize_sample(sample: Sample, tokenizer, response_token_ids: list[int]) -> Sample:
    assert len(sample.loss_mask) == len(response_token_ids)
    assert len(sample.rollout_log_probs) == len(response_token_ids)

    sample.response_length = len(response_token_ids)
    sample.response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    sample.trajectory = tokenizer.decode(sample.tokens, skip_special_tokens=False)
    if sample.status is None or sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample

