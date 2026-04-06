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
from .exp_bank import ExperienceBank, TrajectoryExperience
from ..prompts import render_system_prompt
from ..utils import init_env_client, parse_last_xml, get_experience_bank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# custom generate function (--custom-generate-function-path)
# ---------------------------------------------------------------------------


async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample | List[Sample]:
    """
    multi-turn rollout function for one prompt with subagent verifier
    """
    # Prepare for rollout engine
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = sampling_params.copy()
    sampling_params["no_stop_trim"] = True  # ChatML wrapping requires <|im_end|> in output

    # Prepare for environment resources
    config = getattr(args, "custom_config")
    step = compute_rollout_step(args, sample.metadata["rollout_id"])
    enable_verify = _enable_verify(config, step)
    print(f"[DEBUG] enable verify {enable_verify}")
    experience_bank = get_experience_bank(config) if enable_verify else None
    max_turn = int(config["max_env_turns"])
    max_subagent_turn = int(config.get("max_subagent_turns", 10))
    env_nums = config["env_nums"]
    env_port = config["env_port_base"] + random.randint(0, env_nums - 1)
    env_address = f"http://localhost:{env_port}"

    data_source = sample.metadata["data_source"]
    sample.metadata["role"] = "mainagent"

    # get init obs from env
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

    # start agent loop
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
            enable_subagent=enable_verify,
            experience_bank=experience_bank,
        )
    finally:
        await asyncio.to_thread(env.close)

    all_samples = [main_sample] + subagent_samples
    main_sample.subagent_trajectories = [s.trajectory for s in subagent_samples]
    if evaluation:
        return main_sample

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
    experience_bank: ExperienceBank | None,
) -> tuple[Sample, list[Sample]]:
    if enable_subagent:
        mode = "execution"
    else:
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
    max_repeat = 3
    done = False
    experience = TrajectoryExperience(task=task, action_list=[], obs_list=[])
    subagent_count = 0

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

        if parsed.type is None:
            obs = env.invalid_action_obs
            done = False
            turn += 1  # avoid infinite loop if turn is not incremented
        elif parsed.type == "subagent" and enable_subagent and experience_bank is not None:
            obs, reward, done, sub_sample, sub_turn = await run_subagent(
                args=args,
                parent_sample=sample,
                task=task,
                tokenizer=tokenizer,
                url=url,
                sampling_params=sampling_params,
                experience_bank=experience_bank,
                experience=experience,
            )
            subagent_samples.append(sub_sample)
            rewards.append(reward)
            subagent_count += 1
            turn += 1
        elif parsed.type == "action":
            step_output = await asyncio.to_thread(env.step, parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            rewards.append(reward)
            turn += 1
            if data_source == "searchqa" and enable_subagent and turn % 10:
                obs += "\n# I should now invoke the verifier agent to check my execution, with <subagent> verify and calibrate by current execution </subagent>"
            experience.update(action=parsed.content, obs=obs)
            if _should_stop_on_repeat(experience.action_list, max_repeat):
                logger.info(f"Detected {max_repeat} repeated actions, terminating trajectory early")
                sample.status = Sample.Status.COMPLETED
                break
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
        outcome_reward = rewards[-1] / 100 if rewards and rewards[-1] > 0 else 0.0
    else:
        outcome_reward = rewards[-1] if rewards else 0.0
    if subagent_count == 0:
        subagent_bonus = 0.0
    elif subagent_count <= 5:
        subagent_bonus = 0.1
    else:
        subagent_bonus = -0.1
    sample.outcome_reward = outcome_reward
    sample.subagent_bonus = subagent_bonus
    sample.reward = outcome_reward + subagent_bonus
    sample.rewards = rewards
    sample.metadata["turn"] = turn
    sample.metadata["subagent_count"] = subagent_count
    sample.metadata["experience"] = experience
    logger.info(
        "\033[32m#### outcome_reward: %s, subagent_bonus: %s, reward: %s, done: %s, turn: %s, subagent_count: %s, token budget: %s\033[0m",
        sample.outcome_reward,
        sample.subagent_bonus,
        sample.reward,
        sample.status,
        turn,
        subagent_count,
        budget,
    )
    main_sample = _finalize(sample, tokenizer, response_token_ids)
    return main_sample, subagent_samples


async def run_subagent(
    args: Any,
    parent_sample: Sample,
    task: str,
    tokenizer,
    url: str,
    sampling_params: dict,
    *,
    experience_bank: ExperienceBank,
    experience: TrajectoryExperience,
) -> tuple[str, float, bool, Sample, int]:
    # prepare for subagent loop
    subagent_system_prompt = render_system_prompt(
        env_name=parent_sample.metadata["data_source"],
        mode="verifier",
        task=task,
    )
    sub_messages = [{"role": "system", "content": subagent_system_prompt}]
    main_traj = experience.recent_act_obs_traj if parent_sample.metadata["data_source"] in ["webshop", "searchqa"] else experience.act_obs_traj
    retrieved_context = experience_bank.retrieve(task, top_k=3)
    user_prompt = (f"# Trajectory to be verified\n"
                   f"{main_traj}\n\n"
                   f"# Retrieved summarized experience patterns from experience bank\n"
                   f"{retrieved_context if retrieved_context else ''}\n")
    sub_messages.append({"role": "user", "content": user_prompt})


    # Initialize subagent sample with TITO tracking
    sub_sample = deepcopy(parent_sample)
    sub_sample.prompt = sub_messages
    sub_sample.metadata["role"] = "subagent"

    prompt_ids = tokenizer.apply_chat_template(sub_messages, tokenize=True, add_generation_prompt=True)
    response_token_ids = _init_sample_state(sub_sample, prompt_ids)
    budget = args.rollout_max_context_len - len(prompt_ids)
    resp_text, new_token_ids, new_log_probs = await _generate_one_turn(
        input_ids=sub_sample.tokens,
        url=url,
        sampling_params=sampling_params,
        budget=budget,
    )
    _append_to_sample(sub_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)

    feedback = resp_text
    finalized = _finalize(sub_sample, tokenizer, response_token_ids)
    return feedback, 0.0, False, finalized, 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_process(samples: List[Sample]):
    # set reward
    assert samples[0].metadata["role"] == "mainagent", "Assert the first sample come from main agent"
    outcome_reward = samples[0].outcome_reward
    subagent_bonus = samples[0].subagent_bonus
    for i in range(1, len(samples)):
        samples[i].reward = outcome_reward
        samples[i].outcome_reward = outcome_reward
        samples[i].subagent_bonus = subagent_bonus
    return samples

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
    recent_actions = action_list[-max_repeat:]
    return len(set(recent_actions)) == 1

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

def _enable_verify(config: dict, step: int | None) -> bool:
    warmup_step = int(config.get("warmup_step", 30))
    if step is None:
        return True
    return step >= warmup_step
