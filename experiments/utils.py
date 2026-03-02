"""
A simple tool parser

support format:
<action> act_str </action>
<subagent> act_str </subagent>
"""
import re
import yaml
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from slime.utils.metric_utils import compute_rollout_step

import shlex
import subprocess
from collections.abc import Mapping
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass
class ParseResult:
    type: str # in ("action", "subagent")
    content: str # action/subagent input/conclusion

def tool_parser(response: str) -> ParseResult | None:
    pattern = r"<(action|subagent|conclusion)>(.*?)</\1>"
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if not matches:
        return None
    m = matches[-1]
    return ParseResult(type=m.group(1), content=m.group(2).strip())

def _is_success(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return value in {1, 100}
    return 0

def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    assert rollout_extra_metrics is not None
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
    return False


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
            f.write(f"{sample.metadata.get('task_desc')}\n")
            f.write(sample.response)
            if hasattr(sample, "subagent_responses"):
                for i, sub_resp in enumerate(sample.subagent_responses):
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
    for flag, value in arg_map.items():
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