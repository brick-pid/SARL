import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


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


def _run_shell(command: str, *, env: dict[str, str], check: bool) -> int:
    print(f"$ {command}")
    completed = subprocess.run(command, shell=True, env=env, check=False)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def _load_model_args(cfg: DictConfig, *, env: dict[str, str]) -> list[str]:
    script_path = os.path.join(get_original_cwd(), str(cfg.model.script))
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


def _detect_nvlink_references(env: dict[str, str]) -> int:
    command = "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l"
    completed = subprocess.run(command, shell=True, env=env, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return 0

    match = re.search(r"\d+", completed.stdout)
    return int(match.group(0)) if match else 0

def _resolve_reward_postprocess_path(cfg: DictConfig) -> str | None:
    reward_type = OmegaConf.select(cfg, "custom.reward_norm", default=None)
    if reward_type is None:
        return None
    mapping = {
        "v1": "experiments.rewards.post_process_rewards_v1",
        "v2": "experiments.rewards.post_process_rewards_v2",
        "v3": "experiments.rewards.post_process_rewards_v3",
    }
    if reward_type in mapping:
        return mapping[reward_type]

    # Allow direct function path
    return str(reward_type).strip()

@hydra.main(version_base="1.3", config_path="hydra_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    repo_dir = get_original_cwd()
    os.chdir(repo_dir)

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.runtime.env_vars.items()})

    nvlink_count = _detect_nvlink_references(env)
    has_nvlink = 1 if nvlink_count > 0 else 0
    print(f"REPO_DIR: {repo_dir}")
    print(f"HAS_NVLINK: {has_nvlink} (detected {nvlink_count} NVLink references)")

    for command in cfg.runtime.cleanup_commands:
        _run_shell(str(command), env=env, check=not cfg.runtime.cleanup_ignore_errors)

    ray_start_cmd = ["ray", "start"] + _render_args(cfg.gpu.ray_start)
    _run_command(ray_start_cmd, env=env)

    exp_name = str(cfg.paths.exp_name)
    dump_dir = str(cfg.misc.dump_base_dir)
    if bool(cfg.misc.dump_append_exp_name):
        dump_dir = os.path.join(dump_dir, exp_name)
    custom_config_path = cfg.paths.repo_dir + "/experiments/hydra_conf/custom/" + f"{cfg.custom.name}.yaml"
    model_args = _load_model_args(cfg, env=env)
    reward_postprocess_path = _resolve_reward_postprocess_path(cfg)

    train_args: list[str] = []
    train_args.extend(_render_args(cfg.gpu.resources_cli))
    train_args.extend(model_args)
    train_args.extend(_render_args(cfg.checkpoint.cli))
    train_args.extend(_render_args(cfg.rollout.cli))
    train_args.extend(_render_args(cfg.optimizer.cli))
    train_args.extend(_render_args(cfg.algo.cli))
    train_args.extend(_render_args(cfg.logging.cli))
    train_args.extend(_render_args(cfg.gpu.perf_cli))
    train_args.extend(_render_args(cfg.eval.cli))
    train_args.extend(_render_args(cfg.sglang.cli))
    train_args.extend(_render_args(cfg.misc.cli))
    train_args.extend(["--dump-details", dump_dir])
    train_args.extend(["--custom-config-path", custom_config_path])
    train_args.extend(["--custom-generate-function-path", "experiments.generate.generate"])
    train_args.extend(["--custom-rollout-log-function-path", "experiments.utils.log_rollout_data"])
    train_args.extend(["--custom-eval-rollout-log-function-path", "experiments.utils.log_eval_rollout_data"])
    if reward_postprocess_path is not None: train_args.extend(["--custom-reward-post-process-path", reward_postprocess_path])

    breakpoint()
    submit_cmd = [
        "ray",
        "job",
        "submit",
        f"--address={cfg.gpu.ray_job.address}",
        f"--runtime-env-json={json.dumps(cfg.gpu.ray_job.runtime_env, separators=(',', ':'))}",
        "--",
        str(cfg.runtime.python_executable),
        str(cfg.runtime.train_entrypoint),
        *train_args,
    ]
    _run_command(submit_cmd, env=env)


if __name__ == "__main__":
    main()
