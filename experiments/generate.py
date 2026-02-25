"""
Multi-turn, sub-agent generate function for Gym-style environments.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .env import GymEnv
from .utils import tool_parser
from .prompts import (
    env2system_prompt,
    subagent_prompt_patch,
    subagent_system_prompt
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# custom generate function (--custom-generate-function-path)
# ---------------------------------------------------------------------------

async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample] | Sample:
    """
    Multi-turn generate function for Gym-style environments.

    Uses post() to call SGLang /generate endpoint.

    Tracks tokens, loss_mask, and logprobs manually (TITO):
      - Model-generated tokens: loss_mask=1, logprobs from SGLang
      - Environment observation tokens: loss_mask=0, logprobs=0.0

    When the model emits <subagent>task</subagent>, a subagent is spawned
    to run a shorter agent loop on the env. The subagent's conclusion is
    fed back to the main agent as an observation, and the subagent's Sample
    is collected for training alongside the main agent's Sample.

    Returns list[Sample]: [main_sample, *subagent_samples].
    """
    config = getattr(args, "custom_config")
    max_turns = config["max_turns"]
    env_addresses = config["env_addresses"]

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # --- Prepare training setup ---
    task_id = int(sample.prompt)  # --input-key task_id stores value here
    data_source = sample.metadata["data_source"]
    env_address = env_addresses[data_source]

    cumulative_reward = 0.0
    step_rewards: list[float] = []
    num_turns = 0
    sampling_params = sampling_params.copy()
    sampling_params["no_stop_trim"] = True  # ChatML wrapping requires <|im_end|> in output
    sample.metadata["role"] = "mainagent"
    sample.metadata["task_id"] = task_id
    subagent_samples: list[Sample] = []

    # --- Environment Init ---
    env = GymEnv(env_name=data_source, address=env_address)
    obs, info = await env.reset(task_id=task_id)
    done = False
    action_list = []
    obs_list = []
    task = obs # use init obs as task description

    # --- Build prompt: system + obs as user turn ---
    system_prompt = env2system_prompt[data_source]
    if config.get("enable_subagent", False):
        system_prompt += "\n\n" + subagent_prompt_patch
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs},
    ]
    prompt_ids = tokenizer.apply_chat_template(chat_messages, tokenize=True, add_generation_prompt=True)

    # Pre-compute ChatML turn boundary tokens for obs wrapping.
    # Model output already ends with <|im_end|> (no_stop_trim=True),
    # so _turn_pre starts with \n (not <|im_end|>).
    _turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    _turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    # Token-level accumulators (TITO: no retokenization)
    all_token_ids: list[int] = list(prompt_ids)
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []

    # --- Compute token budget ---
    budget = args.rollout_max_context_len - len(prompt_ids)

    for turn_idx in range(max_turns):
        # --- Model generates action ---
        cur_params = sampling_params.copy()
        cur_params["max_new_tokens"] = budget

        payload = {"input_ids": all_token_ids, "sampling_params": cur_params, "return_logprob": True}
        output = await post(url, payload)

        # --- Extract tokens & logprobs (TITO) ---
        raw_logprobs = output["meta_info"]["output_token_logprobs"]
        new_token_ids = [item[1] for item in raw_logprobs]
        new_log_probs = [item[0] for item in raw_logprobs]

        resp_text = output["text"]

        # Accumulate model output (loss_mask=1)
        all_token_ids.extend(new_token_ids)
        response_token_ids.extend(new_token_ids)
        loss_mask.extend([1] * len(new_token_ids))
        rollout_log_probs.extend(new_log_probs)

        budget -= len(new_token_ids)

        # --- Parse action ---
        parsed = tool_parser(resp_text)
        if parsed is None:
            obs = "The task is not completed yet, you need to take an action in current environment."
            reward, done, info = 0.0, False, {}
            step_rewards.append(reward)
        elif parsed.type == "subagent":
            breakpoint()
            obs, reward, done, sub_sample = await subagent_generate(args=args, parent_sample=sample, task=task, subtask=parsed.content,
                env=env, tokenizer=tokenizer, url=url, sampling_params=sampling_params, config=config)
            subagent_samples.append(sub_sample)
            cumulative_reward += reward
            step_rewards.append(reward)
        elif parsed.type == "action":
            obs, reward, done, info = await env.step(parsed.content)
            cumulative_reward += reward
            step_rewards.append(reward)
        else:
            raise ValueError(f"Unrecognized tool response type: {parsed.type}")
        if parsed is not None:
            action_list.append(parsed.content)
            obs_list.append(obs)
        # --- Wrap observation in ChatML user turn (loss_mask=0) ---
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        all_token_ids.extend(turn_ids)
        response_token_ids.extend(turn_ids)
        loss_mask.extend([0] * len(turn_ids))
        rollout_log_probs.extend([0.0] * len(turn_ids))
        num_turns = turn_idx + 1

        budget -= len(turn_ids)
        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if done:
            sample.status = Sample.Status.COMPLETED
            break

    await env.close()
    # --- Finalize main sample ---
    if env.env_name == "sciworld":
        cumulative_reward /= 100
    sample.reward = cumulative_reward
    logger.info(f"#### reward: {sample.reward}, done: {sample.status}, turn: {num_turns}, token budget: {budget}")
    sample.metadata["num_turns"] = num_turns
    sample.metadata["step_rewards"] = step_rewards
    sample.metadata["action_list"] = action_list
    sample.metadata["obs_list"] = obs_list
    main_sample = _finalize(sample, tokenizer, all_token_ids,
                            response_token_ids, loss_mask, rollout_log_probs)
    if evaluation:
        return main_sample
    all_samples = [main_sample] + subagent_samples
    all_samples = _post_process(samples=all_samples, reward_strategy="simple")
    return all_samples

async def subagent_generate(args: Any, parent_sample: Sample, subtask: str, env: GymEnv, tokenizer, url: str, sampling_params: dict, config: dict) -> tuple[str, float, bool, Sample]:
    """
    Spawn a subagent that runs a shorter agent loop on the same env.
    The subagent will train with the main agent.

    Returns:
        obs(conclusion): str to feed back into the main agent as observation
        reward: reward collected by subagent, during interacting with env
        done: whether the environment reached a terminal state
        sample: subagent sample with full TITO data for training
    """
    sub_max_turns = config["subagent_max_turns"]
    task = parent_sample.metadata["task"]
    user_prompt = build_subagent_user_prompt(task=task, subtask=subtask, sample=parent_sample)
    sub_messages = [
        {"role": "system", "content": subagent_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    sub_sample = deepcopy(parent_sample)
    sub_sample.prompt = sub_messages
    sub_sample.metadata["role"] = "subagent"

    prompt_ids = tokenizer.apply_chat_template(sub_messages, tokenize=True, add_generation_prompt=True)

    # avoid retokenization drift
    _turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    _turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    all_token_ids: list[int] = list(prompt_ids)
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []

    budget = args.rollout_max_context_len - len(prompt_ids)

    obs = ""
    cumulative_reward = 0.0
    done = False

    breakpoint()
    for turn_idx in range(sub_max_turns):
        cur_params = sampling_params.copy()
        cur_params["max_new_tokens"] = budget

        payload = {"input_ids": all_token_ids, "sampling_params": cur_params, "return_logprob": True}
        output = await post(url, payload)

        raw_logprobs = output["meta_info"]["output_token_logprobs"]
        new_token_ids = [item[1] for item in raw_logprobs]
        new_log_probs = [item[0] for item in raw_logprobs]

        resp_text = output["text"]

        # Accumulate model output (loss_mask=1)
        all_token_ids.extend(new_token_ids)
        response_token_ids.extend(new_token_ids)
        loss_mask.extend([1] * len(new_token_ids))
        rollout_log_probs.extend(new_log_probs)

        budget -= len(new_token_ids)

        parsed = tool_parser(resp_text)
        if parsed is None:
            obs = "The task is not completed yet, you need to take an action in current environment."
            reward, done, info = 0.0, False, {}
        elif parsed.type == "action":
            obs, reward, done, info = await env.step(parsed.content)
            cumulative_reward += reward
            if turn_idx == sub_max_turns - 2:
                obs += "\n\nNext turn is your last turn. You have to give the final conclusion in the next turn, and return control to main agent."
        elif parsed.type == "conclusion":
            obs = parsed.content
        else:
            raise ValueError(f"Unrecognized tool response type from subagent: {parsed.type}")

        # Encode env observation as ChatML user turn (loss_mask=0)
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        all_token_ids.extend(turn_ids)
        response_token_ids.extend(turn_ids)
        loss_mask.extend([0] * len(turn_ids))
        rollout_log_probs.extend([0.0] * len(turn_ids))

        budget -= len(turn_ids)
        if budget <= 0:
            sub_sample.status = Sample.Status.TRUNCATED
            break
        if done or (parsed is not None and parsed.type == "conclusion"):
            sub_sample.status = Sample.Status.COMPLETED
            break

    # We don't set reward for subagent here, currently, we use main agent outcome reward as subagent reward signal (reward_strategy="simple").
    # In the future, we can explore more sophisticated reward assignment strategy and set subagent reward here.
    finalized = _finalize(sub_sample, tokenizer, all_token_ids,
                          response_token_ids, loss_mask, rollout_log_probs)
    return obs, cumulative_reward, done, finalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_process(samples: List[Sample], reward_strategy: str = "simple"):
    """
    Reward Assignment Strategy
    """
    # ------ Simple Reward Strategy ------
    if reward_strategy == "simple":
        assert samples[0].metadata["role"] == "mainagent", "Assert the first sample come from main agent"
        outcome_reward = samples[0].reward
        for i in range(1, len(samples)):
            samples[i].reward = outcome_reward
    else:
        raise NotImplementedError(f"Not support reward strategy: {reward_strategy}")
    return samples
    

def _finalize(
    sample: Sample, tokenizer, all_token_ids: list[int], response_token_ids: list[int], loss_mask: list[int], rollout_log_probs: list[float]) -> Sample:
    """Pack token-level tracking data into the Sample."""
    # alignment checks
    assert len(loss_mask) == len(response_token_ids), f"loss_mask length {len(loss_mask)} != response_token_ids length {len(response_token_ids)}"
    assert len(rollout_log_probs) == len(response_token_ids), f"rollout_log_probs length {len(rollout_log_probs)} != response_token_ids length {len(response_token_ids)}"

    sample.tokens = all_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs

    # Keep full response text (model + obs/tool/env)
    sample.response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    if sample.status is None or sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample

def build_subagent_user_prompt(task, subtask, sample):
    action_list = sample.metadata["action_list"]
    obs_list = sample.metadata["obs_list"]
    history_str = "".join(f"Action: {a}\nObservation: {o}\n" for a, o in zip(action_list, obs_list))
    prompt = f"""# Input Context\n## Task\n{task}\n## Subtask\n{subtask}\n## Main Agent History\n{history_str}""".strip()
    return prompt