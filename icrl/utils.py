from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Mapping

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def render_cli_args(arg_map: Mapping[str, object]) -> list[str]:
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


def run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def load_model_args(cfg: DictConfig, *, env: dict[str, str]) -> list[str]:
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


def dump_resolved_custom_config(cfg: DictConfig) -> str:
    custom_cfg = OmegaConf.select(cfg, "custom", default=None)
    if custom_cfg is None:
        raise ValueError("Missing `custom` config in Hydra config.")

    resolved_custom = OmegaConf.to_container(custom_cfg, resolve=True)

    exp_dir = OmegaConf.select(cfg, "paths.exp_dir", default=None)
    if exp_dir is None:
        exp_dir = HydraConfig.get().runtime.output_dir

    custom_config_path = str(exp_dir) + "/custom_config.yaml"
    with open(custom_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_custom, f, sort_keys=False)
    return custom_config_path


def parse_last_xml(text: str, tag: str) -> str | None:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None
