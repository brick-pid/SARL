"""
Multi-turn, sub-agent generate function for Gym-style environments.
"""

from __future__ import annotations

import random
import logging
from copy import deepcopy
from typing import Any, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# from .env import GymEnv
from .utils import tool_parser, init_env_client
from .prompts import (
    render_main_system_prompt,
    render_subagent_system_prompt,
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
    max_turn = int(config["max_env_turns"])
    env_nums = config["env_nums"]
    env_port = config["env_port_base"] + random.randint(0, env_nums-1)
    env_address = f"http://localhost:{env_port}"

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # --- Prepare training setup ---
    task_id = int(sample.prompt)  # --input-key task_id stores value here
    data_source = sample.metadata["data_source"]

    cumulative_reward = 0.0
    sampling_params = sampling_params.copy()
    sampling_params["no_stop_trim"] = True  # ChatML wrapping requires <|im_end|> in output
    sample.metadata["role"] = "mainagent"
    sample.metadata["task_id"] = task_id
    subagent_samples: list[Sample] = []

    # --- Environment Init ---
    env = init_env_client(env_name=data_source, env_addr=env_address)
    env.reset(task_id)
    obs = env.observe()
        
    done = False
    task = obs # use init obs as task description
    sample.metadata["task_desc"] = task

    # --- Build prompt/message history state ---
    system_prompt = env.conversation_start[0]["value"] #TODO: add subagent patch
    chat_messages = [{"role": "system", "content": system_prompt}, {"role": "assistant", "content": env.conversation_start[1]["value"]}, {"role": "user", "content": obs}]
    prompt_ids = tokenizer.apply_chat_template(chat_messages, tokenize=True, add_generation_prompt=True)
    # Model output already ends with <|im_end|> (no_stop_trim=True),
    # so turn_pre starts with "\n" (not <|im_end|>).
    _turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    _turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    # Token-level accumulators (TITO: no retokenization)
    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    response_token_ids: list[int] = []

    # --- Compute token budget ---
    budget = args.rollout_max_context_len - len(prompt_ids)

    turn = 0
    recent_actions: list[str] = []
    MAX_REPEAT = 3

    while True:
        # --- Model generates action ---
        cur_params = sampling_params.copy()
        cur_params["max_new_tokens"] = budget

        payload = {"input_ids": sample.tokens, "sampling_params": cur_params, "return_logprob": True}
        output = await post(url, payload)

        # --- Extract tokens & logprobs (TITO) ---
        raw_logprobs = output["meta_info"]["output_token_logprobs"]
        new_token_ids = [item[1] for item in raw_logprobs]
        new_log_probs = [item[0] for item in raw_logprobs]

        resp_text = output["text"]

        # Accumulate model output (loss_mask=1)
        _append_to_sample(sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)

        budget -= len(new_token_ids)

        # --- Parse action ---
        parsed = tool_parser(resp_text)

        # --- Repeated action detection ---
        current_action = parsed.content if parsed else resp_text
        if len(recent_actions) >= MAX_REPEAT - 1 and all(a == current_action for a in recent_actions[-(MAX_REPEAT-1):]):
            logger.info(f"Detected {MAX_REPEAT} repeated actions, terminating trajectory early")
            sample.status = Sample.Status.COMPLETED
            break
        recent_actions.append(current_action)

        if parsed is None:
            obs = "The task is not completed yet. Think more carefully about the environment."
            reward, done, info = 0.0, False, {}
            turn += 1 # avoid infinite loop if turn is not incremented
        elif parsed.type == "subagent":
            obs, reward, done, sub_sample, sub_turn = await subagent_generate(args=args, parent_sample=sample, task=task, subtask=parsed.content,
                                                                    env=env, tokenizer=tokenizer, url=url, 
                                                                    sampling_params=sampling_params, config=config)
            subagent_samples.append(sub_sample)
            cumulative_reward += reward
            turn += sub_turn
        elif parsed.type == "action":
            step_output = env.step(parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            cumulative_reward += reward
            turn += 1
        else:
            # unknown parsed.type
            break
        # --- Wrap observation in ChatML user turn (loss_mask=0) ---
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        _append_to_sample(sample, response_token_ids, turn_ids, [0.0] * len(turn_ids), loss_mask_val=0)

        budget -= len(turn_ids)
        if budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        if turn >= max_turn or done:
            sample.status = Sample.Status.COMPLETED
            break

    try:
        env.close()
    except Exception as e:
        print(f"Error during closing env: {e}")
    # --- Finalize main sample ---
    if data_source == "sciworld":
        if cumulative_reward < 0:
            cumulative_reward = 0
        cumulative_reward /= 100
    sample.reward = cumulative_reward
    logger.info(f"#### reward: {sample.reward}, done: {sample.status}, turn: {turn}, token budget: {budget}")
    sample.metadata["turn"] = turn
    main_sample = _finalize(sample, tokenizer, response_token_ids)
    if evaluation:
        main_sample.subagent_responses = [s.response for s in subagent_samples]
        return main_sample
    all_samples = [main_sample] + subagent_samples
    all_samples = _post_process(samples=all_samples, reward_strategy="simple")
    return all_samples

async def subagent_generate(args: Any, parent_sample: Sample, task: str, subtask: str,
                            env, tokenizer, url: str, sampling_params: dict, config: dict) -> tuple[str, float, bool, Sample, int]:
    """
    Spawn a subagent that runs a shorter agent loop on the same env.
    The subagent will train with the main agent.

    Returns:
        obs(conclusion): str to feed back into the main agent as observation
        reward: reward collected by subagent, during interacting with env
        done: whether the environment reached a terminal state
        sample: subagent sample with full TITO data for training
        turn: number of turns the subagent took
    """
    max_turn = int(config["max_subagent_turns"])
    subagent_system_prompt = render_subagent_system_prompt(
        env_name=env.env_name,
        subtask=subtask,
    )
    sub_messages = [{"role": "system", "content": subagent_system_prompt}]

    sub_sample = deepcopy(parent_sample)
    sub_sample.prompt = sub_messages
    sub_sample.metadata["role"] = "subagent"

    prompt_ids = tokenizer.apply_chat_template(sub_messages, tokenize=True, add_generation_prompt=True)
    # avoid retokenization drift
    _turn_pre = tokenizer.encode("\n<|im_start|>user\n", add_special_tokens=False)
    _turn_post = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    sub_sample.tokens = list(prompt_ids)
    sub_sample.loss_mask = []
    sub_sample.rollout_log_probs = []
    response_token_ids: list[int] = []

    budget = args.rollout_max_context_len - len(prompt_ids)

    obs = ""
    cumulative_reward = 0.0
    done = False

    action_list = []
    obs_list = []
    recent_actions: list[str] = []
    MAX_REPEAT = 3

    turn = 0
    while True:
        turn += 1
        cur_params = sampling_params.copy()
        cur_params["max_new_tokens"] = budget

        payload = {"input_ids": sub_sample.tokens, "sampling_params": cur_params, "return_logprob": True}
        output = await post(url, payload)

        raw_logprobs = output["meta_info"]["output_token_logprobs"]
        new_token_ids = [item[1] for item in raw_logprobs]
        new_log_probs = [item[0] for item in raw_logprobs]

        resp_text = output["text"]

        # Accumulate model output (loss_mask=1)
        _append_to_sample(sub_sample, response_token_ids, new_token_ids, new_log_probs, loss_mask_val=1)

        budget -= len(new_token_ids)

        parsed = tool_parser(resp_text)

        # --- Repeated action detection ---
        current_action = parsed.content if parsed else resp_text
        if len(recent_actions) >= MAX_REPEAT - 1 and all(a == current_action for a in recent_actions[-(MAX_REPEAT-1):]):
            logger.info(f"Subagent detected {MAX_REPEAT} repeated actions, terminating early")
            sub_sample.status = Sample.Status.COMPLETED
            break
        recent_actions.append(current_action)

        if not parsed:
            obs = "Your response is not valid. Use <action> ... </action> to interact with environment."
            reward, done, info = 0, False, {}
        elif parsed.type == "action" and "COMPLETED" in parsed.content:
            break
        elif parsed.type == "action":
            step_output = env.step(parsed.content)
            obs, reward, done = step_output.state, step_output.reward, step_output.done
            cumulative_reward += reward
            action_list.append(parsed.content)
            obs_list.append(obs)
        else:
            # raise ValueError(f"Unrecognized tool response type from subagent: {parsed.type}")
            break

        # Encode env observation as ChatML user turn (loss_mask=0)
        obs_ids = tokenizer.encode(obs, add_special_tokens=False)
        turn_ids = _turn_pre + obs_ids + _turn_post
        _append_to_sample(sub_sample, response_token_ids, turn_ids, [0.0] * len(turn_ids), loss_mask_val=0)

        budget -= len(turn_ids)
        if budget <= 0:
            sub_sample.status = Sample.Status.TRUNCATED
            break
        if turn >= max_turn or done:
            sub_sample.status = Sample.Status.COMPLETED
            break

    obs_str = "Subagent Actions and Observations\n"
    for i in range(len(action_list)):
        obs_str += f"{action_list[i]} -> {obs_list[i]}\n"
    obs = obs_str
    # We don't set reward for subagent here, currently, we use main agent outcome reward as subagent reward signal (reward_strategy="simple").
    # In the future, we can explore more sophisticated reward assignment strategy and set subagent reward here.
    finalized = _finalize(sub_sample, tokenizer, response_token_ids)
    return obs, cumulative_reward, done, finalized, turn


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


def _finalize(sample: Sample, tokenizer, response_token_ids: list[int]) -> Sample:
    """Pack token-level tracking data into the Sample."""
    # alignment checks
    assert len(sample.loss_mask) == len(response_token_ids)
    assert len(sample.rollout_log_probs) == len(response_token_ids)

    sample.response_length = len(response_token_ids)
    sample.response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    sample.trajectory = tokenizer.decode(sample.tokens, skip_special_tokens=False)
    if sample.status is None or sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED
    return sample
