from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from experiments.envs.math_env import MathEnvClient
from experiments.utils import init_env_client

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
from .schema import EpisodeRecord, RoundRecord
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
    verifier_max_new_tokens = int(config.get("verifier_max_new_tokens", 1024))
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

        episode = EpisodeRecord(task=task, data_source=data_source)
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
                max_rounds=max_rounds,
                max_turn=max_turn,
                max_repeat=max_repeat,
                previous_critic=critic_history[-1] if critic_history else None,
                critic_history=critic_history,
            )

            verifier_sample, verifier_pred_success = await _run_judge_round(
                args=args,
                base_sample=sample,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                role="verifier",
                task=task,
                trajectory_summary=exec_sample.metadata["trajectory_summary"],
                executor_reward=exec_reward,
                round_id=round_id,
                max_rounds=max_rounds,
                max_new_tokens=verifier_max_new_tokens,
            )
            exec_success = is_success_reward(exec_reward)
            verifier_reward = 1.0 if verifier_pred_success == exec_success else 0.0

            round_record = RoundRecord(
                round_id=round_id,
                executor_sample=exec_sample,
                executor_reward=exec_reward,
                verifier_sample=verifier_sample,
                verifier_pred_success=verifier_pred_success,
                verifier_reward=verifier_reward,
            )
            episode.rounds.append(round_record)

            if exec_success or round_id >= max_rounds:
                break

            critic_sample, critic_text = await _run_judge_round(
                args=args,
                base_sample=sample,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                role="critic",
                task=task,
                trajectory_summary=exec_sample.metadata["trajectory_summary"],
                executor_reward=exec_reward,
                round_id=round_id,
                max_rounds=max_rounds,
                max_new_tokens=critic_max_new_tokens,
            )
            round_record.critic_sample = critic_sample
            critic_history.append(critic_text)

        _assign_rewards(episode)
        all_samples = _flatten_episode(episode)
        if evaluation:
            return episode.rounds[-1].executor_sample if episode.rounds else sample
        return all_samples
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
    max_rounds: int,
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

    system_prompt = render_role_prompt(
        env_name=data_source,
        role="executor",
        task=task,
    )
    context_block = _build_executor_user_context(task=task, init_obs=init_obs, previous_critic=previous_critic)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_block},
    ]

    exec_sample = clone_role_sample(base_sample, role="executor", round_id=round_id, prompt=messages)
    exec_sample.metadata["task_desc"] = task
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
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
    exec_sample.metadata["trajectory_summary"] = _build_trajectory_summary(
        task=task,
        actions=action_history,
        observations=observation_history,
        normalized_reward=outcome_reward,
    )
    finalized = finalize_sample(exec_sample, tokenizer, response_token_ids)
    return finalized, outcome_reward, done


async def _run_judge_round(
    *,
    args: Any,
    base_sample: Sample,
    tokenizer,
    url: str,
    sampling_params: dict,
    role: str,
    task: str,
    trajectory_summary: str,
    executor_reward: float,
    round_id: int,
    max_rounds: int,
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

    resp_text, new_token_ids, new_log_probs = await generate_one_turn(
        input_ids=role_sample.tokens,
        url=url,
        sampling_params=sampling_params,
        budget=max_new_tokens,
    )
    append_to_sample(role_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)
    finalized = finalize_sample(role_sample, tokenizer, response_token_ids)

    if role == "verifier":
        verdict = (parse_last_xml(resp_text, "verdict") or "").strip().lower()
        pred_success = verdict == "correct"
        return finalized, pred_success

    critic_text = parse_last_xml(resp_text, "critic") or resp_text.strip()
    return finalized, critic_text


def _assign_rewards(episode: EpisodeRecord) -> None:
    rounds = episode.rounds
    for idx, record in enumerate(rounds):
        record.executor_sample.reward = record.executor_reward
        record.executor_sample.outcome_reward = record.executor_reward

        record.verifier_sample.reward = record.verifier_reward
        record.verifier_sample.outcome_reward = record.verifier_reward

        if record.critic_sample is not None:
            next_exec_reward = rounds[idx + 1].executor_reward if idx + 1 < len(rounds) else record.executor_reward
            critic_reward = next_exec_reward - record.executor_reward
            record.critic_reward = critic_reward
            record.critic_sample.reward = critic_reward
            record.critic_sample.outcome_reward = critic_reward


def _flatten_episode(episode: EpisodeRecord) -> list[Sample]:
    samples: list[Sample] = []
    for record in episode.rounds:
        samples.append(record.executor_sample)
        samples.append(record.verifier_sample)
        if record.critic_sample is not None:
            samples.append(record.critic_sample)
    return samples


def _build_executor_user_context(*, task: str, init_obs: str, previous_critic: str | None) -> str:
    sections = [f"# Task\n{task}"]
    if previous_critic:
        sections.append("# Core Tips From Previous Critique\n" + previous_critic)
    sections.append("# Initial Observation\n" + init_obs)
    return "\n\n".join(sections)


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
        problem = sample.prompt[0]["content"] if isinstance(sample.prompt, list) else sample.prompt
        label = sample.label
        env = MathEnvClient(problem=problem, label=label)
        return env, env.observe()

    task_id = int(sample.prompt)
    env = await asyncio.to_thread(init_env_client, env_name=data_source, env_addr=env_address)
    await asyncio.to_thread(env.reset, task_id)
    init_obs = await asyncio.to_thread(env.observe)
    return env, init_obs


def _normalize_outcome_reward(data_source: str, reward: float) -> float:
    if data_source == "sciworld":
        return reward / 100 if reward > 0 else 0.0
    return float(reward)
