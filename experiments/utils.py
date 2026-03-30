"""
Utility functions for experiments.
"""
import re
import yaml
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from slime.utils.metric_utils import compute_rollout_step

import shlex
import subprocess
from collections.abc import Mapping
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from .generates.exp_bank import Experience, ExperienceBank

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

def _is_success(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value in {1, 100}
    return 0

def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
    if args.custom_config['generate'] == "verify":
        exp_bank = get_experience_bank(args.custom_config)
        experiences = []
        for s in samples:
            if s.metadata["role"] == "mainagent" and s.reward == 1.0:
                experiences.append(s.metadata["experience"])
        # breakpoint()
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


def get_experience_bank(config: dict) -> ExperienceBank | None:
    global _EXPERIENCE_BANK

    bank_dir = config.get("exp_dir")
    if _EXPERIENCE_BANK is not None:
        return _EXPERIENCE_BANK

    bank = ExperienceBank(bank_dir)
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
        subset = sample.metadata.get('subset', sample.metadata.get("data_source"))
        output_path = basedir / "rollouts" / f"eval{step}_{subset}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(f"reward: {sample.reward}\n")
            f.write(f"rewards: {sample.rewards}\n")
            f.write(f"{sample.metadata.get('task_desc')}\n")
            f.write(sample.trajectory)
            if hasattr(sample, "subagent_trajectories"):
                for i, sub_resp in enumerate(sample.subagent_trajectories):
                    f.write(f"### subagent response {i}\n")
                    f.write(sub_resp)
            f.write("\n==============\n")
    
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
