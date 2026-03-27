"""
Multi-turn, sub-agent generate function for Gym-style environments.
"""

from __future__ import annotations

import asyncio
import logging
import random
from copy import deepcopy
from typing import Any, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.types import Sample

from ..envs.math_env import MathEnvClient
from ..prompts import render_system_prompt
from ..utils import init_env_client, parse_last_xml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# custom generate function (--custom-generate-function-path)
# ---------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample | List[Sample]:
    """
    Multi-turn generate function for Gym-style environments.

    Uses post() to call SGLang /generate endpoint.

    Tracks tokens, loss_mask, and logprobs manually (TITO):
      - Model-generated tokens: loss_mask=1, logprobs from SGLang
      - Environment observation tokens: loss_mask=0, logprobs=0.0

    When the model emits <subagent>task</subagent>, a subagent is spawned
    to run a shorter agent loop on the env. The subagent's observation is
    fed back to the main agent as an observation, and the subagent's Sample
    is collected for training alongside the main agent's Sample.
    """
    config = getattr(args, "custom_config")
    max_turn = int(config["max_env_turns"])
    max_subagent_turn = int(config.get("max_subagent_turns", 10))
    env_nums = config["env_nums"]
    env_port = config["env_port_base"] + random.randint(0, env_nums - 1)
    env_address = f"http://localhost:{env_port}"
    step = compute_rollout_step(args, sample.metadata["rollout_id"])
    phase = _resolve_strategy_phase(config=config, step=step)
    enable_subagent = phase["enable_subagent"]
    train_subagent = phase["train_subagent"]

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sampling_params = sampling_params.copy()
    sampling_params["no_stop_trim"] = True  # ChatML wrapping requires <|im_end|> in output

    data_source = sample.metadata["data_source"]
    sample.metadata["role"] = "mainagent"
    sample.metadata["step"] = step

    env, init_obs = await _open_env_for_sample(
        sample=sample,
        data_source=data_source,
        env_address=env_address,
    )
    if "AVAILABLE ACTIONS" in init_obs:
        task = init_obs.split("AVAILABLE ACTIONS")[0]
    else:
        task = init_obs
    sample.metadata["task_desc"] = task

    try:
        main_sample, subagent_samples = await run_main_loop(
            args=args,
            sample=sample,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            url=url,
            env=env,
            task=task,
            init_obs=init_obs,
            data_source=data_source,
            max_turn=max_turn,
            max_subagent_turn=max_subagent_turn,
            enable_subagent=enable_subagent,
            step=step,
        )
    finally:
        await asyncio.to_thread(env.close)

    all_samples = [main_sample] + subagent_samples
    # breakpoint()
    if evaluation:
        main_sample.subagent_trajectories = [s.trajectory for s in subagent_samples]
        return main_sample

    if not train_subagent and main_sample.reward != 1:
        for sample in all_samples[1:]:
            sample.remove_sample = True

    return _post_process(samples=all_samples)


async def run_main_loop(
    args: Any,
    sample: Sample,
    sampling_params: dict,
    *,
    tokenizer,
    url: str,
    env,
    task: str,
    init_obs: str,
    data_source: str,
    max_turn: int,
    max_subagent_turn: int,
    enable_subagent: bool,
    step: int | None = None,
) -> tuple[Sample, list[Sample]]:
    mode = "single"
    system_prompt = render_system_prompt(env_name=data_source, mode=mode, task=task)
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.append({"role": "user", "content": init_obs})

    prompt_ids = tokenizer.apply_chat_template(chat_messages, tokenize=True, add_generation_prompt=True)
    turn_pre, turn_post = _build_chat_turn_markers(tokenizer)
    response_token_ids = _init_sample_state(sample, prompt_ids)
    budget = args.rollout_max_context_len - len(prompt_ids)

    turn = 0
    rewards: list[float] = []
    subagent_samples: list[Sample] = []
    history: list[str] = [f"Initial Observation:\n{init_obs}"]
    recent_actions: list[str] = []
    max_repeat = 4
    done = False

    while True:
        resp_text, new_token_ids, new_log_probs = await _generate_one_turn(
            input_ids=sample.tokens,
            url=url,
            sampling_params=sampling_params,
            budget=budget,
        )
        _append_to_sample(sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)
        budget -= len(new_token_ids)

        parsed = env.parse_response(resp_text)
        recent_actions.append(parsed.content)
        if _should_stop_on_repeat(recent_actions, max_repeat):
            logger.info(f"Detected {max_repeat} repeated actions, terminating trajectory early")
            sample.status = Sample.Status.COMPLETED
            break

        if parsed.type is None:
            obs = env.invalid_action_obs
            done = False
            turn += 1  # avoid infinite loop if turn is not incremented
        elif parsed.type == "subagent" and enable_subagent:
            obs, reward, done, sub_sample, sub_turn = await run_subagent_loop(
                args=args,
                parent_sample=sample,
                task=task,
                env=env,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                max_turn=max_subagent_turn,
                history=history,
            )
            subagent_samples.append(sub_sample)
            rewards.append(reward)
            # history.append(f"Agent Response:\n{resp_text}\n\nObservation:\n{obs}")
            # turn += sub_turn # don't count verifier turns
        elif parsed.type == "action":
            step_output = await asyncio.to_thread(env.step, parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            rewards.append(reward)
            turn += 1
            main_obs = obs
            history.append(f"<action>{parsed.content}</action>\n<observation>\n{obs}</observation>")

            if enable_subagent and not done and (turn >= 10 and turn % 10 == 0 or _should_stop_on_repeat(recent_actions, max_repeat-1)):
                verifier_obs, verifier_reward, verifier_done, verifier_sample, _ = await run_subagent_loop(
                    args=args,
                    parent_sample=sample,
                    task=task,
                    env=env,
                    tokenizer=tokenizer,
                    url=url,
                    sampling_params=sampling_params,
                    max_turn=max_subagent_turn,
                    history=history,
                )
                subagent_samples.append(verifier_sample)
                rewards.append(verifier_reward)
                done = verifier_done
                if verifier_obs:
                    obs = f"{main_obs}\n\n[Verifier Feedback]\n{verifier_obs}"
        else:
            break

        budget -= _append_observation_turn(
            sample=sample,
            response_tokens=response_token_ids,
            tokenizer=tokenizer,
            turn_pre=turn_pre,
            turn_post=turn_post,
            obs=obs,
        )
        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if turn >= max_turn or done:
            sample.status = Sample.Status.COMPLETED
            break

    if data_source == "sciworld":
        sample.reward = rewards[-1] / 100 if rewards and rewards[-1] > 0 else 0.0
    else:
        sample.reward = rewards[-1] if rewards else 0.0
    sample.rewards = rewards
    sample.metadata["turn"] = turn
    if step is not None:
        sample.metadata["step"] = step
    logger.info(f"\033[32m#### reward: {sample.reward}, done: {sample.status}, turn: {turn}, token budget: {budget}\033[0m")
    main_sample = _finalize(sample, tokenizer, response_token_ids)
    return main_sample, subagent_samples


async def run_subagent_loop(
    args: Any,
    parent_sample: Sample,
    task: str,
    env,
    tokenizer,
    url: str,
    sampling_params: dict,
    *,
    max_turn: int,
    history: List | None = None,
) -> tuple[str, float, bool, Sample, int]:
    """
    Spawn a subagent verifier that runs a shorter agent loop on the same env.
    The subagent will train with the main agent.

    Returns:
        obs: str to feed back into the main agent as observation
        reward: reward collected by subagent, during interacting with env
        done: whether the environment reached a terminal state
        sample: subagent sample with full TITO data for training
        turn: number of turns the subagent took
    """
    subagent_system_prompt = render_system_prompt(
        env_name=parent_sample.metadata["data_source"],
        mode="verifier",
        task=task,
    )
    sub_messages = [{"role": "system", "content": subagent_system_prompt}]
    history_text = "\n\n".join(history[-10:]) if history else "No prior history."
    sub_messages.append({"role": "user", "content": f"# History segment to be verified\n{history_text}"})

    sub_sample = deepcopy(parent_sample)
    sub_sample.prompt = sub_messages
    sub_sample.metadata["role"] = "subagent"

    prompt_ids = tokenizer.apply_chat_template(sub_messages, tokenize=True, add_generation_prompt=True)
    turn_pre, turn_post = _build_chat_turn_markers(tokenizer)
    response_token_ids = _init_sample_state(sub_sample, prompt_ids)
    budget = args.rollout_max_context_len - len(prompt_ids)

    obs = ""
    rewards: list[float] = []
    done = False

    action_list: list[str] = []
    obs_list: list[str] = []
    recent_actions: list[str] = []
    max_repeat = 3

    turn = 0
    while True:
        turn += 1
        resp_text, new_token_ids, new_log_probs = await _generate_one_turn(
            input_ids=sub_sample.tokens,
            url=url,
            sampling_params=sampling_params,
            budget=budget,
        )
        _append_to_sample(sub_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)
        budget -= len(new_token_ids)

        parsed = env.parse_response(resp_text)
        recent_actions.append(parsed.content)
        if _should_stop_on_repeat(recent_actions, max_repeat):
            logger.info(f"Subagent detected {max_repeat} repeated actions, terminating early")
            sub_sample.status = Sample.Status.COMPLETED
            break

        if parsed.type == "action":
            step_output = await asyncio.to_thread(env.step, parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            rewards.append(reward)
            action_list.append(parsed.content)
            obs_list.append(obs)
        else:
            sub_sample.status = Sample.Status.COMPLETED
            break

        budget -= _append_observation_turn(
            sample=sub_sample,
            response_tokens=response_token_ids,
            tokenizer=tokenizer,
            turn_pre=turn_pre,
            turn_post=turn_post,
            obs=obs,
        )
        if budget <= 0:
            sub_sample.status = Sample.Status.TRUNCATED
            break
        if turn >= max_turn or done:
            sub_sample.status = Sample.Status.COMPLETED
            break
    
    # prepare feedback as obs
    feedback = parse_last_xml(resp_text, tag="feedback") or resp_text
    act_obs = ""
    for i in range(len(action_list)):
        act_obs += f"{action_list[i]} -> {obs_list[i]}\n"
    feedback += f"\nAction Execution during Verification\n{act_obs}"

    # We don't set reward for subagent here. Currently, we use main agent
    # outcome reward as subagent reward signal (reward_strategy="simple").
    finalized = _finalize(sub_sample, tokenizer, response_token_ids)
    final_reward = rewards[-1] if rewards else 0.0
    return feedback, final_reward, done, finalized, turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post_process(samples: List[Sample]):
    # set reward
    assert samples[0].metadata["role"] == "mainagent", "Assert the first sample come from main agent"
    outcome_reward = samples[0].reward
    for i in range(1, len(samples)):
        samples[i].reward = outcome_reward
    return samples


def _resolve_strategy_phase(config: dict, step: int) -> dict:
    schedule = config.get("strategy_schedule")
    if not schedule:
        raise ValueError("custom_config.strategy_schedule is required")
    last_until_step = None
    for phase in schedule:
        until_step = phase.get("until_step")
        if until_step is not None:
            until_step = int(until_step)
            if last_until_step is not None and until_step <= last_until_step:
                raise ValueError("strategy_schedule until_step must be strictly increasing")
            last_until_step = until_step
            if step < until_step:
                return phase
        else:
            return phase
    return schedule[-1]


async def _open_env_for_sample(sample: Sample, data_source: str, env_address: str):
    if data_source == "math":
        problem = sample.prompt[0]["content"] if isinstance(sample.prompt, list) else sample.prompt
        label = sample.label
        env = MathEnvClient(problem=problem, label=label)
        sample.metadata["task_id"] = None
        return env, env.observe()

    task_id = int(sample.prompt)  # --input-key task_id stores value here
    sample.metadata["task_id"] = task_id
    env = await asyncio.to_thread(init_env_client, env_name=data_source, env_addr=env_address)
    await asyncio.to_thread(env.reset, task_id)
    init_obs = await asyncio.to_thread(env.observe)
    return env, init_obs


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _build_chat_turn_markers(tokenizer) -> tuple[list[int], list[int]]:
    # Model output already ends with <|im_end|> (no_stop_trim=True),
    # so turn_pre starts with "\n" (not <|im_end|>).
    turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    return turn_pre, turn_post


def _init_sample_state(sample: Sample, prompt_ids: list[int]) -> list[int]:
    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    return []


async def _generate_one_turn(
    *,
    input_ids: list[int],
    url: str,
    sampling_params: dict,
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


def _append_observation_turn(
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
    _append_to_sample(sample, response_tokens, turn_ids, [0.0] * len(turn_ids), loss_mask_val=0)
    return len(turn_ids)


def _should_stop_on_repeat(action_list: list[str], max_repeat: int) -> bool:
    if len(action_list) < max_repeat:
        return False
    recent_action = action_list[-max_repeat:]
    if len(set(recent_action)) == 1:
        return True
    return False


def _finalize(sample: Sample, tokenizer, response_token_ids: list[int]) -> Sample:
    """Pack token-level tracking data into the Sample."""
    assert len(sample.loss_mask) == len(response_token_ids)
    assert len(sample.rollout_log_probs) == len(response_token_ids)

    sample.response_length = len(response_token_ids)
    sample.response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    sample.trajectory = tokenizer.decode(sample.tokens, skip_special_tokens=False)
    if sample.status is None or sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample
