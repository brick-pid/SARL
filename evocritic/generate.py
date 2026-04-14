from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from experiments.envs.math_env import MathEnvClient
from experiments.utils import init_env_client

from .experience_bank import load_experience_bank
from .experience_bank import Trajectory
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
    retrieval_top_k_tasks = int(config.get("experience_bank_top_k_tasks", 5))
    critic_use_retrieval = bool(config.get("critic_use_retrieval", True))
    critic_use_retrieved_trajectories = bool(config.get("critic_use_retrieved_trajectories", True))
    experience_bank = load_experience_bank(config) if critic_use_retrieval else None

    env_nums = config["env_nums"]
    env_port = config["env_port_base"] + random.randint(0, env_nums - 1)
    env_address = f"http://localhost:{env_port}"

    data_source = sample.metadata["data_source"]

    env, init_obs = await _open_env_for_sample(sample, data_source, env_address)
    try:
        task = init_obs.split("AVAILABLE ACTIONS")[0] if "AVAILABLE ACTIONS" in init_obs else init_obs
        task_id = _resolve_task_id(sample=sample, data_source=data_source, task=task)
        sample.metadata["task_id"] = task_id
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
                trajectory_summary=exec_sample.metadata["trajectory_summary_verifier"],
                round_id=round_id,
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

            retrieved_context_text = ""
            if experience_bank is not None:
                try:
                    retrieved_context_text = experience_bank.retrieve(
                        task,
                        top_k_tasks=retrieval_top_k_tasks,
                    ).to_text(include_trajectories=critic_use_retrieved_trajectories).strip()
                except Exception:
                    logger.exception("Failed to retrieve experience bank context for task")
            critic_sample, critic_text = await _run_judge_round(
                args=args,
                base_sample=sample,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                role="critic",
                task=task,
                trajectory_summary=exec_sample.metadata["trajectory_summary"],
                round_id=round_id,
                max_new_tokens=critic_max_new_tokens,
                retrieved_context_text=retrieved_context_text,
            )
            round_record.critic_sample = critic_sample
            critic_history.append(critic_text)

        _assign_rewards(episode)
        all_samples = _flatten_episode(episode)
        if evaluation:
            if not episode.rounds:
                return sample
            final_sample = episode.rounds[-1].executor_sample
            final_sample.metadata["round_rewards"] = [record.executor_reward for record in episode.rounds]
            final_sample.metadata["round_successes"] = [
                is_success_reward(record.executor_reward) for record in episode.rounds
            ]
            return final_sample
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
    exec_sample.metadata["trajectory_steps"] = [
        {"action": action, "observation": observation}
        for action, observation in zip(action_history, observation_history, strict=True)
    ]
    trajectory = Trajectory(
        task_desc=task,
        turn=turn,
        reward=outcome_reward,
        steps=exec_sample.metadata["trajectory_steps"],
    )
    exec_sample.metadata["trajectory_summary"] = trajectory.to_text(header="trajectory:")
    exec_sample.metadata["trajectory_summary_verifier"] = trajectory.to_text(
        header="trajectory:",
        view="verifier",
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
    round_id: int,
    max_new_tokens: int,
    retrieved_context_text: str = "",
) -> tuple[Sample, bool | str]:
    system_prompt = render_role_prompt(
        env_name=base_sample.metadata["data_source"],
        role=role,
        task=task,
    )
    trajectory_section_title = "# Current Trajectory Needs Verify" if role == "verifier" else "# Current Trajectory Needs Critique"
    user_lines = [
        f"# Current Task\n{task}",
        f"{trajectory_section_title}\n{trajectory_summary}",
    ]
    retrieved_context_text = retrieved_context_text.strip()
    if role == "critic" and retrieved_context_text:
        user_lines.append(f"# Relevant Experience for Reference \n{retrieved_context_text}")
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
    finalized.metadata["critique_text"] = critic_text
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
            record.critic_sample.metadata["reward_prev"] = record.executor_reward
            record.critic_sample.metadata["reward_after"] = next_exec_reward
            record.critic_sample.metadata["reward_diff"] = critic_reward


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

def _resolve_task_id(*, sample: Sample, data_source: str, task: str) -> str:
    existing_task_id = sample.metadata.get("task_id")
    if existing_task_id:
        return str(existing_task_id)

    prompt = sample.prompt
    if data_source != "math" and not isinstance(prompt, list):
        return f"{data_source}::{prompt}"

    task_hash = hashlib.sha1(task.encode("utf-8")).hexdigest()
    return f"{data_source}::{task_hash}"
