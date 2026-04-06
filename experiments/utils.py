"""
Utility functions for experiments.
"""
import logging
import re
import asyncio
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess

import httpx
import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .generates.exp_bank import Experience, ExperienceBank
from .prompts import render_system_prompt
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.metric_utils import compute_rollout_step

# environments
from contextlib import contextmanager
import time
from .envs.controller.types import ActionFormat
from .envs import (
    AcademiaEnvClient,
    AlfWorldEnvClient,
    BabyAIEnvClient,
    MazeEnvClient,
    MovieEnvClient,
    SciworldEnvClient,
    SheetEnvClient,
    SqlGymEnvClient,
    TextCraftEnvClient,
    TodoEnvClient,
    WeatherEnvClient,
    WebarenaEnvClient,
    WebshopEnvClient,
    WordleEnvClient,
    SearchQAEnvClient,
)

_EXPERIENCE_BANK: ExperienceBank | None = None
logger = logging.getLogger(__name__)

def _is_success(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value in {1, 100}
    return 0


def _group_samples_for_summary(samples):
    grouped = defaultdict(list)
    for sample in samples:
        if sample.metadata.get("role") != "mainagent":
            continue
        assert sample.group_index is not None, "mainagent sample.group_index must not be None"
        grouped[sample.group_index].append(sample)

    summary_groups = []
    for group_index, group_samples in grouped.items():
        tasks = {sample.metadata.get("task_desc") for sample in group_samples}
        # assert len(tasks) == 1, f"group {group_index} contains inconsistent task_desc values: {tasks}"
        rewards = np.asarray([sample.reward for sample in group_samples], dtype=float)
        if np.any(rewards > 0.5):
            summary_groups.append((group_index, group_samples))
    return summary_groups


def _serialize_group_for_summary(group_samples) -> str:
    lines = []
    for idx, sample in enumerate(group_samples, start=1):
        trajectory_experience = sample.metadata.get("experience")
        assert trajectory_experience is not None, "mainagent sample.metadata['experience'] must not be None"
        data_source = sample.metadata.get("data_source")
        trajectory_text = (
            trajectory_experience.recent_act_obs_traj
            if data_source in ["webshop", "searchqa"]
            else trajectory_experience.act_obs_traj
        )
        lines.append(f"[trajectory_{idx}]")
        lines.append(f"reward: {sample.reward}")
        lines.append(trajectory_text)
        lines.append("")
    return "\n".join(lines).strip()


async def _summarize_group_async(
    args,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    group_index,
    group_samples,
) -> Experience | None:
    logger.info("Summarizing experience group %s with %s samples", group_index, len(group_samples))
    task = group_samples[0].metadata["task_desc"]
    env_name = group_samples[0].metadata["data_source"]
    instruction_prompt = render_system_prompt(env_name=env_name, mode="summarize", task=task)
    user_message = (
        "# Trajectory Group\n"
        "Below are trajectories sampled from the same prompt group. Analyze them jointly.\n\n"
        f"{_serialize_group_for_summary(group_samples)}"
    )

    state = GenerateState(args)
    tokenizer = state.tokenizer
    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(prompt_ids) > 32768:
        logger.warning(
            "Skipping experience summary for group %s in %s because prompt length %s exceeds limit %s. task=%r",
            group_index,
            env_name,
            len(prompt_ids),
            32768,
            task,
        )
        return None

    sampling_params = state.sampling_params.copy()
    sampling_params["temperature"] = 0.0
    sampling_params["top_p"] = 1.0
    sampling_params["max_new_tokens"] = int(args.custom_config.get("summary_max_new_tokens", 2048))
    sampling_params["no_stop_trim"] = True
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
    }
    url = "http://127.0.0.1:38001/generate"
    max_retries = int(args.custom_config.get("summary_max_retries", 60))

    async with semaphore:
        retry_count = 0
        while retry_count < max_retries:
            response = None
            try:
                response = await client.post(url, json=payload or {})
                response.raise_for_status()
                content = await response.aread()
                try:
                    output = json.loads(content)
                except json.JSONDecodeError:
                    output = content.decode() if isinstance(content, bytes) else content
                break
            except Exception as e:
                retry_count += 1
                response_text = e.response.text if isinstance(e, httpx.HTTPStatusError) else None
                logger.info(
                    f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
                )
                if retry_count >= max_retries:
                    logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                    raise
                await asyncio.sleep(1)
            finally:
                if response is not None:
                    await response.aclose()
    summary = output["text"].strip()
    if not summary:
        raise ValueError("Summarization output is empty")
    return Experience(task=task, summary=summary)


async def _summarize_groups_async(args, summary_groups) -> list[Experience]:
    if not summary_groups:
        return []

    concurrency = int(args.custom_config.get("summary_concurrency", len(summary_groups)))
    concurrency = max(1, min(concurrency, len(summary_groups)))
    max_connections = max(1, concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_connections),
        timeout=httpx.Timeout(None),
    ) as client:
        tasks = [
            _summarize_group_async(args, client, semaphore, group_index, group_samples)
            for group_index, group_samples in summary_groups
        ]
        results = await asyncio.gather(*tasks)
        return [exp for exp in results if exp is not None]


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
    _save_train_rollout_trajectories(rollout_id, args, samples)
    if args.custom_config['generate'] == "verify":
        exp_bank = get_experience_bank(args.custom_config)
        summary_groups = _group_samples_for_summary(samples)
        experiences = asyncio.run(_summarize_groups_async(args, summary_groups))
        exp_bank.add_experiences(experiences)
    success_count = 0
    subset_count = defaultdict(int)
    subset_success_count = defaultdict(int)
    for sample in samples:
        success = _is_success(sample.reward)
        success_count += success
        # subset
        subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
        subset_count[subset] += 1
        subset_success_count[subset] += success

    rollout_extra_metrics["rollout/success_rate"] = success_count / len(samples)
    for subset in subset_count:
        rollout_extra_metrics[f"rollout/{subset}/success_rate"] = subset_success_count[subset] / subset_count[subset]

    # --- Turn metrics (mainagent only) ---
    subset_turns = defaultdict(list)
    all_turns = []
    for sample in samples:
        if sample.metadata.get("role") != "mainagent":
            continue
        turn = sample.metadata.get("turn")
        if turn is None:
            continue
        all_turns.append(turn)
        subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
        subset_turns[subset].append(turn)

    if all_turns:
        rollout_extra_metrics["rollout/turn/mean"] = np.mean(all_turns).item()
        rollout_extra_metrics["rollout/turn/max"] = np.max(all_turns).item()
        rollout_extra_metrics["rollout/turn/min"] = np.min(all_turns).item()
        for subset, turns in subset_turns.items():
            rollout_extra_metrics[f"rollout/{subset}/turn/mean"] = np.mean(turns).item()
            rollout_extra_metrics[f"rollout/{subset}/turn/max"] = np.max(turns).item()
            rollout_extra_metrics[f"rollout/{subset}/turn/min"] = np.min(turns).item()

    return False


def _save_train_rollout_trajectories(rollout_id, args, samples) -> None:
    step = compute_rollout_step(args, rollout_id)
    basedir = Path(args.custom_config["exp_dir"])
    output_path = basedir / "rollouts_train" / f"train_{step}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            if sample.metadata.get("role") != "mainagent":
                continue
            _write_sample_trajectory(f, idx, sample)


def _write_sample_trajectory(f, sample_idx, sample) -> None:
    f.write(f"sample_idx: {sample_idx}\n")
    f.write("role: mainagent\n")
    f.write(f"reward: {sample.reward}\n")
    subset = sample.metadata.get("subset", sample.metadata.get("data_source"))
    if subset is not None:
        f.write(f"subset: {subset}\n")
    if "task_desc" in sample.metadata:
        f.write(f"task_desc: {sample.metadata['task_desc']}\n")

    trajectory = str(getattr(sample, "trajectory", sample.response))
    f.write(trajectory)
    if not trajectory.endswith("\n"):
        f.write("\n")

    subagent_trajectories = getattr(sample, "subagent_trajectories", None) or []
    for i, sub_traj in enumerate(subagent_trajectories):
        f.write(f"### subagent trajectory {i}\n")
        sub_traj = str(sub_traj)
        f.write(sub_traj)
        if not sub_traj.endswith("\n"):
            f.write("\n")

    f.write("\n==============\n")


def get_experience_bank(config: dict) -> ExperienceBank | None:
    global _EXPERIENCE_BANK

    bank_dir = config.get("exp_dir")
    if _EXPERIENCE_BANK is not None:
        return _EXPERIENCE_BANK

    bank = ExperienceBank(
        bank_dir,
        resume_experience_bank_path=config.get("resume_experience_bank_path"),
    )
    _EXPERIENCE_BANK = bank
    return bank

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
            subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
            subset_count[subset] += 1
            subset_success_count[subset] += success
        _log_helper(rollout_id, args, samples)
        extra_metrics[f"eval/{dataset_name}/success_rate"] = success_count / len(samples)
        for subset in subset_count:
            extra_metrics[f"eval/{dataset_name}/{subset}/success_rate"] = subset_success_count[subset] / subset_count[subset]

        # --- Turn metrics (mainagent only) ---
        subset_turns = defaultdict(list)
        all_turns = []
        for sample in samples:
            if sample.metadata.get("role") != "mainagent":
                continue
            turn = sample.metadata.get("turn")
            if turn is None:
                continue
            all_turns.append(turn)
            subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
            subset_turns[subset].append(turn)

        if all_turns:
            extra_metrics[f"eval/{dataset_name}/turn/mean"] = np.mean(all_turns).item()
            extra_metrics[f"eval/{dataset_name}/turn/max"] = np.max(all_turns).item()
            extra_metrics[f"eval/{dataset_name}/turn/min"] = np.min(all_turns).item()
            for subset, turns in subset_turns.items():
                extra_metrics[f"eval/{dataset_name}/{subset}/turn/mean"] = np.mean(turns).item()
                extra_metrics[f"eval/{dataset_name}/{subset}/turn/max"] = np.max(turns).item()
                extra_metrics[f"eval/{dataset_name}/{subset}/turn/min"] = np.min(turns).item()

    return False

def _log_helper(rollout_id, args, samples):
    """
    save rollout trajectory and metrics
    """
    # hack save rollout
    step = compute_rollout_step(args, rollout_id)
    basedir = Path(args.custom_config['exp_dir'])
    metric_path = basedir / "rollouts" / f"eval_{step}.metrics"
    for sample in samples:
        if sample.metadata.get("role") != "mainagent":
            continue
        subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
        output_path = basedir / "rollouts" / f"eval{step}_{subset}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"rewards: {sample.rewards}\n")
            _write_sample_trajectory(f, sample.metadata.get("task_id", "unknown"), sample)
    
    with open(metric_path, "a") as f:
        rewards = []
        subset_rewards = {}
        for sample in samples:
            rewards.append(sample.reward)
            subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
            if subset not in subset_rewards:
                subset_rewards[subset] = [sample.reward]
            else:
                subset_rewards[subset].append(sample.reward)
        metric_str = f"STEP: {step}\n"
        metric_str += f"--overall rewards: {sum(rewards)}/{len(rewards)}\n"
        for k, v in subset_rewards.items():
            metric_str += f"----subset rewards for {k}: {sum(v)/len(v)}\n"
        f.write(metric_str)
        f.write("\n")


def _render_args(arg_map: Mapping[str, object]) -> list[str]:
    args: list[str] = []
    for key, value in arg_map.items():
        flag = f"--{key.replace('_', '-')}"
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args

def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)

def _load_model_args(cfg: DictConfig, *, env: dict[str, str]) -> list[str]:
    script_path = cfg.model.script_path
    bash_command = f'set -euo pipefail; source "{script_path}"; printf "%s\\n" "${{MODEL_ARGS[@]}}"'
    completed = subprocess.run(
        ["bash", "-lc", bash_command],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not args:
        raise ValueError(f"No MODEL_ARGS were loaded from script: {cfg.model.script}")
    return args

def _dump_resolved_custom_config(cfg: DictConfig) -> str:
    custom_cfg = OmegaConf.select(cfg, "custom", default=None)
    if custom_cfg is None:
        raise ValueError("Missing `custom` config in Hydra config.")

    resolved_custom = OmegaConf.to_container(custom_cfg, resolve=True)

    exp_dir = OmegaConf.select(cfg, "paths.exp_dir", default=None)
    if exp_dir is None:
        exp_dir = HydraConfig.get().runtime.output_dir

    exp_dir = str(exp_dir)
    custom_config_path = exp_dir + "/custom_config.yaml"
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_custom, f, sort_keys=False)
    return custom_config_path


# Environment client, modified from AgentGym
# NOTE: only REACT_XML action format is supported for the experiments pipeline.
def init_env_client(env_name, env_addr, max_retries=3, action_format="react_xml"):
    # task_name - task dict
    envclient_classes = {
        "webshop": WebshopEnvClient,
        "alfworld": AlfWorldEnvClient,
        "babyai": BabyAIEnvClient,
        "sciworld": SciworldEnvClient,
        "textcraft": TextCraftEnvClient,
        "webarena": WebarenaEnvClient,
        "sqlgym": SqlGymEnvClient,
        "maze": MazeEnvClient,
        "wordle": WordleEnvClient,
        "weather": WeatherEnvClient,
        "todo": TodoEnvClient,
        "movie": MovieEnvClient,
        "sheet": SheetEnvClient,
        "academia": AcademiaEnvClient,
        "searchqa": SearchQAEnvClient,
    }
    # select task according to the name
    envclient_class = envclient_classes.get(env_name)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {env_name}")
    retry = 0
    while True:
        try:
            env_client = envclient_class(env_server_base=env_addr, data_len=1, timeout=2400, action_format=action_format)
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server {env_addr}, retrying...({retry}/{max_retries})")
            if retry > max_retries:
                raise e
            time.sleep(5)
    return env_client

def parse_last_xml(str, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, str, re.DOTALL)
    if matches:
        return matches[-1].strip()
    else:
        return None
