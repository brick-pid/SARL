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
        try:
            output = await post(url, payload)
        except Exception as e:
            sample.status = Sample.Status.TRUNCATED
            logger.warning(f"ERROR During Generation: {type(e).__name__}: {e}")
            break

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
            sample.status = Sample.Status.ABORTED
            break

        if parsed.type == "subagent":
            trajectory = tokenizer.decode(response_token_ids, skip_special_tokens=False)
            obs, reward, done, sub_sample = await subagent_generate(
                args=args,
                parent_sample=sample,
                task=parsed.content,
                trajectory=trajectory,
                env=env,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                config=config,
            )
            subagent_samples.append(sub_sample)
            cumulative_reward += reward
            step_rewards.append(reward)

            if done:
                sample.status = Sample.Status.COMPLETED
                break
        elif parsed.type == "action":
            try:
                obs, reward, done, info = await env.step(parsed.content)
                cumulative_reward += reward
                step_rewards.append(reward)
            except Exception as e:
                sample.status = Sample.Status.TRUNCATED
                logger.warning(f"ERROR During Env: {type(e).__name__}: {e}")
                break

            if done:
                sample.status = Sample.Status.COMPLETED
                break
        else:
            sample.status = Sample.Status.ABORTED
            break

        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        # --- Wrap observation in ChatML user turn (loss_mask=0) ---
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        all_token_ids.extend(turn_ids)
        response_token_ids.extend(turn_ids)
        loss_mask.extend([0] * len(turn_ids))
        rollout_log_probs.extend([0.0] * len(turn_ids))

        budget -= len(turn_ids)
        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        num_turns = turn_idx + 1

    await env.close()
    # --- Finalize main sample ---
    sample.reward = cumulative_reward
    sample.metadata["gym_num_turns"] = num_turns
    sample.metadata["gym_step_rewards"] = step_rewards
    main_sample = _finalize(sample, tokenizer, all_token_ids,
                            response_token_ids, loss_mask, rollout_log_probs)
    if evaluation:
        return main_sample
    all_samples = [main_sample] + subagent_samples
    all_samples = _post_process(samples=all_samples, reward_strategy="simple")
    return all_samples

async def subagent_generate(
    *,
    args: Any,
    parent_sample: Sample,
    task: str,
    trajectory: str,
    env: GymEnv,
    tokenizer,
    url: str,
    sampling_params: dict,
    config: dict,
) -> tuple[str, float, bool, Sample]:
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
    user_prompt = f"# Partial Trajectory:\n{trajectory}"
    sub_messages = [
        {"role": "system", "content": subagent_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    sub_sample = deepcopy(parent_sample)
    sub_sample.prompt = sub_messages
    sub_sample.reward = None
    sub_sample.status = None
    sub_sample.metadata = {
        **(parent_sample.metadata or {}),
        "role": "subagent",
        "subagent_task": task,
    }

    prompt_ids = tokenizer.apply_chat_template(sub_messages, tokenize=True, add_generation_prompt=True)

    # avoid retokenization drift
    _turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    _turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    all_token_ids: list[int] = list(prompt_ids)
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []

    budget = args.rollout_max_context_len - len(prompt_ids)

    conclusion = ""
    obs = ""
    cumulative_reward = 0.0
    env_done = False

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
            sub_sample.status = Sample.Status.ABORTED
            break
        if parsed.type == "action":
            obs, reward, done, info = await env.step(parsed.content)
            cumulative_reward += reward
            if done:
                conclusion = obs
                env_done = True
                sub_sample.status = Sample.Status.COMPLETED
                break
        elif parsed.type == "conclusion":
            conclusion = parsed.content
            sub_sample.status = Sample.Status.COMPLETED
            break
        else:
            # subagent should not emit <subagent> (no recursion)
            sub_sample.status = Sample.Status.ABORTED
            break

        if budget <= 0:
            conclusion = obs
            sub_sample.status = Sample.Status.TRUNCATED
            break

        # Encode env observation as ChatML user turn (loss_mask=0)
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        all_token_ids.extend(turn_ids)
        response_token_ids.extend(turn_ids)
        loss_mask.extend([0] * len(turn_ids))
        rollout_log_probs.extend([0.0] * len(turn_ids))

        budget -= len(turn_ids)
        if budget <= 0:
            conclusion = obs
            sub_sample.status = Sample.Status.TRUNCATED
            break
    else:
        # for-loop exhausted sub_max_turns without break
        sub_sample.status = Sample.Status.TRUNCATED

    sub_sample.reward = cumulative_reward
    finalized = _finalize(sub_sample, tokenizer, all_token_ids,
                          response_token_ids, loss_mask, rollout_log_probs)
    return conclusion, cumulative_reward, env_done, finalized


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
    sample: Sample,
    tokenizer,
    all_token_ids: list[int],
    response_token_ids: list[int],
    loss_mask: list[int],
    rollout_log_probs: list[float],
) -> Sample:
    """Pack token-level tracking data into the Sample."""
    sample.tokens = all_token_ids
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.response_length = len(response_token_ids)

    # Decode only model-generated tokens for sample.response
    model_token_ids = [
        tid for tid, mask in zip(response_token_ids, loss_mask) if mask
    ]
    sample.response = tokenizer.decode(model_token_ids, skip_special_tokens=False)

    if sample.status is None or sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample
