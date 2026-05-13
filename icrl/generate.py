from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from .envs import MathEnvClient, init_env_client
from .prompts import render_role_prompt
from .rewards import is_success_reward
from .runtime import (
    append_observation_turn,
    append_to_sample,
    build_chat_turn_markers,
    clone_role_sample,
    finalize_sample,
    generate_one_turn,
    init_sample_state,
    should_stop_on_repeat,
)
from .utils import parse_last_xml

logger = logging.getLogger(__name__)


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False):
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = sampling_params.copy()
    sampling_params["no_stop_trim"] = True

    config = getattr(args, "custom_config")
    max_rounds = int(config.get("max_rounds", 1))
    max_turn = int(config["max_env_turns"])
    max_repeat = int(config.get("max_repeat_actions", 3))
    critic_max_new_tokens = int(config.get("critic_max_new_tokens", 1024))

    env_nums = config["env_nums"]
    env_port = config["env_port_base"] + random.randint(0, env_nums - 1)
    env_address = f"http://localhost:{env_port}"

    data_source = sample.metadata["data_source"]
    sample.metadata["task_id"] = None

    env, init_obs = await _open_env_for_sample(sample, data_source, env_address)
    try:
        task = init_obs.split("AVAILABLE ACTIONS")[0] if "AVAILABLE ACTIONS" in init_obs else init_obs
        sample.metadata["task_desc"] = task

        rollout_samples: list[Sample] = []
        round_successes: list[bool] = []
        critic_history: list[str] = []

        for round_id in range(1, max_rounds + 1):
            exec_sample, exec_reward, exec_done = await _run_executor_round(
                args=args,
                base_sample=sample,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                env=env,
                data_source=data_source,
                task=task,
                init_obs=init_obs,
                round_id=round_id,
                max_turn=max_turn,
                max_repeat=max_repeat,
                previous_critic=critic_history[-1] if critic_history else None,
                critic_history=critic_history,
            )
            exec_success = is_success_reward(exec_reward)
            round_successes.append(exec_success)

            prev_critic = _find_latest_critic(rollout_samples)
            if prev_critic is not None:
                prev_exec_reward = prev_critic.metadata["executor_reward_before_critic"]
                if is_success_reward(exec_reward):
                    prev_critic.reward = 1.0
                    prev_critic.outcome_reward = 1.0
                else:
                    prev_critic.reward = exec_reward - prev_exec_reward
                    prev_critic.outcome_reward = prev_critic.reward

            rollout_samples.append(exec_sample)

            # if evaluation or exec_done or exec_success or round_id >= max_rounds:
            # if exec_done or exec_success or round_id >= max_rounds:
            if exec_success or round_id >= max_rounds:
                break

            critic_sample, critic_text = await _run_judge_round(
                base_sample=sample,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                role="critic",
                task=task,
                trajectory_summary=exec_sample.metadata["trajectory_summary"],
                round_id=round_id,
                max_new_tokens=critic_max_new_tokens,
            )
            critic_sample.reward = 0.0
            critic_sample.outcome_reward = None
            critic_sample.metadata["executor_reward_before_critic"] = exec_reward
            rollout_samples.append(critic_sample)
            critic_history.append(critic_text)

        if evaluation:
            exec_sample.metadata["round_successes"] = round_successes
            return exec_sample
        return rollout_samples
    finally:
        await asyncio.to_thread(env.close)


async def _run_executor_round(
    *,
    args: Any,
    base_sample: Sample,
    tokenizer,
    url: str,
    sampling_params: dict,
    env,
    data_source: str,
    task: str,
    init_obs: str,
    round_id: int,
    max_turn: int,
    max_repeat: int,
    previous_critic: str | None,
    critic_history: list[str],
) -> tuple[Sample, float, bool]:
    if data_source != "math":
        task_id = int(base_sample.prompt)
        await asyncio.to_thread(env.reset, task_id)
    obs = await asyncio.to_thread(env.observe)
    assert obs == init_obs or bool(obs)

    system_prompt = render_role_prompt(env_name=data_source, role="executor", task=task)
    behavior_messages = _build_executor_messages(
        task=task,
        init_obs=init_obs,
        previous_critic=previous_critic,
        system_prompt=system_prompt,
    )
    critic_free_messages = _build_executor_messages(
        task=task,
        init_obs=init_obs,
        previous_critic=None,
        system_prompt=system_prompt,
    )

    exec_sample = clone_role_sample(base_sample, role="executor", round_id=round_id, prompt=behavior_messages)
    exec_sample.metadata["task_desc"] = task
    prompt_ids = tokenizer.apply_chat_template(behavior_messages, tokenize=True, add_generation_prompt=True)
    critic_free_prompt_ids = tokenizer.apply_chat_template(critic_free_messages, tokenize=True, add_generation_prompt=True)
    turn_pre, turn_post = build_chat_turn_markers(tokenizer)
    response_token_ids = init_sample_state(exec_sample, prompt_ids)
    budget = args.rollout_max_context_len - len(prompt_ids)

    rewards: list[float] = []
    action_history: list[str] = []
    observation_history: list[str] = []
    done = False
    turn = 0

    while True:
        resp_text, new_token_ids, new_log_probs = await generate_one_turn(
            input_ids=exec_sample.tokens,
            url=url,
            sampling_params=sampling_params,
            budget=budget,
        )
        append_to_sample(exec_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)
        budget -= len(new_token_ids)

        parsed = env.parse_response(resp_text)
        if parsed.type != "action":
            obs = env.invalid_action_obs
            turn += 1
        else:
            step_output = await asyncio.to_thread(env.step, parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            rewards.append(reward)
            action_history.append(parsed.content)
            observation_history.append(obs)
            turn += 1
            if should_stop_on_repeat(action_history, max_repeat):
                exec_sample.status = Sample.Status.COMPLETED
                break

        budget -= append_observation_turn(
            sample=exec_sample,
            response_tokens=response_token_ids,
            tokenizer=tokenizer,
            turn_pre=turn_pre,
            turn_post=turn_post,
            obs=obs,
        )
        if budget <= 0:
            exec_sample.status = Sample.Status.TRUNCATED
            break
        if turn >= max_turn or done:
            exec_sample.status = Sample.Status.COMPLETED
            break

    outcome_reward = _normalize_outcome_reward(data_source, rewards[-1] if rewards else 0.0)
    exec_sample.reward = outcome_reward
    exec_sample.outcome_reward = outcome_reward
    exec_sample.metadata["turn"] = turn
    exec_sample.metadata["critic_history"] = list(critic_history)
    exec_sample.metadata["previous_critic"] = previous_critic
    exec_sample.metadata["behavior_prompt_ids"] = list(prompt_ids)
    exec_sample.metadata["critic_free_prompt_ids"] = list(critic_free_prompt_ids)
    exec_sample.metadata["trajectory_summary"] = _build_trajectory_summary(
        task=task,
        actions=action_history,
        observations=observation_history,
        normalized_reward=outcome_reward,
    )
    exec_sample.tokens = critic_free_prompt_ids + response_token_ids
    finalized = finalize_sample(exec_sample, tokenizer, response_token_ids)
    return finalized, outcome_reward, done


async def _run_judge_round(
    *,
    base_sample: Sample,
    tokenizer,
    url: str,
    sampling_params: dict,
    role: str,
    task: str,
    trajectory_summary: str,
    round_id: int,
    max_new_tokens: int,
) -> tuple[Sample, bool | str]:
    system_prompt = render_role_prompt(
        env_name=base_sample.metadata["data_source"],
        role=role,
        task=task,
    )
    user_lines = [
        f"# Task\n{task}",
        f"# Trajectory Summary\n{trajectory_summary}",
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]

    role_sample = clone_role_sample(base_sample, role=role, round_id=round_id, prompt=messages)
    role_sample.metadata["task_desc"] = task
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    response_token_ids = init_sample_state(role_sample, prompt_ids)

    # TODO: change critic endpoint by config
    # url = f"http://127.0.0.1:37001/generate"
    resp_text, new_token_ids, new_log_probs = await generate_one_turn(
        input_ids=role_sample.tokens,
        url=url,
        sampling_params=sampling_params,
        budget=max_new_tokens,
    )
    print(f"print critic tokens: {len(new_token_ids)}")
    append_to_sample(role_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)
    finalized = finalize_sample(role_sample, tokenizer, response_token_ids)

    critic_text = parse_last_xml(resp_text, "critic") or resp_text.strip()
    return finalized, critic_text


def _find_latest_critic(samples: list[Sample]) -> Sample | None:
    for sample in reversed(samples):
        if sample.metadata.get("role") == "critic":
            return sample
    return None


def _build_executor_messages(
    *,
    task: str,
    init_obs: str,
    previous_critic: str | None,
    system_prompt: str,
) -> list[dict[str, str]]:
    sections = [f"# Task\n{task}"]
    if previous_critic:
        sections.append("# Core Tips From Previous Critique\n" + previous_critic)
    sections.append("# Initial Observation\n" + init_obs)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _build_trajectory_summary(
    *,
    task: str,
    actions: list[str],
    observations: list[str],
    normalized_reward: float,
) -> str:
    parts = [f"task: {task}", f"reward: {normalized_reward}", "trajectory:"]
    if not actions:
        parts.append(" <empty>")
        return "\n".join(parts)

    for action, observation in zip(actions, observations, strict=True):
        truncated_obs = observation[:800] + ("..." if len(observation) > 800 else "")
        parts.append(f"<action>{action}</action><observation>{truncated_obs}</observation>")
    return "".join(parts).strip()


async def _open_env_for_sample(sample: Sample, data_source: str, env_address: str):
    if data_source == "math":
        problem = _extract_math_problem(sample.prompt)
        label = sample.label
        env = MathEnvClient(problem=problem, label=label)
        sample.metadata["task_id"] = None
        return env, env.observe()

    task_id = int(sample.prompt)
    env = await asyncio.to_thread(init_env_client, env_name=data_source, env_addr=env_address)
    await asyncio.to_thread(env.reset, task_id)
    init_obs = await asyncio.to_thread(env.observe)
    return env, init_obs


def _extract_math_problem(prompt: str | list[dict[str, str]]) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        for message in prompt:
            if isinstance(message.get("content"), str):
                return message["content"]
    raise ValueError("Unsupported math prompt format.")


def _normalize_outcome_reward(data_source: str, reward: float) -> float:
    if data_source == "sciworld":
        return reward / 100 if reward > 0 else 0.0
    return float(reward)
