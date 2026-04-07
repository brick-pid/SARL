import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils import _dump_resolved_custom_config, _load_model_args, _render_args, _run_command

@hydra.main(version_base="1.3", config_path="hydra_conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, cfg.paths.exp_dir + "/config_resoved.yaml")

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.runtime.env_vars.items()})

    ray_start_cmd = ["ray", "start"] + _render_args(cfg.gpu.ray_start)
    _run_command(ray_start_cmd, env=env)

    custom_config_path = _dump_resolved_custom_config(cfg)
    model_args = _load_model_args(cfg, env=env)
    reward_postprocess_path = cfg.custom.reward_process

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
    train_args.extend(["--custom-config-path", custom_config_path])
    if cfg.custom.custom_config.generate == "default":
        train_args.extend(["--custom-generate-function-path", "experiments.generate.generate"])
    elif cfg.custom.custom_config.generate == "verify":
        train_args.extend(["--custom-generate-function-path", "experiments.generates.exp_verifier.generate"])
    elif cfg.custom.custom_config.generate == "verify2":
        train_args.extend(["--custom-generate-function-path", "experiments.generates_v2.exp_verifier.generate"])
    else:
        raise NotImplementedError("[ERROR] generate function path error!")
    train_args.extend(["--custom-rollout-log-function-path", "experiments.utils.log_rollout_data"])
    train_args.extend(["--custom-eval-rollout-log-function-path", "experiments.utils.log_eval_rollout_data"])
    if reward_postprocess_path is not None: train_args.extend(["--custom-reward-post-process-path", reward_postprocess_path])

    submit_cmd = [
        "ray",
        "job",
        "submit",
        f"--address={cfg.gpu.ray_job.address}",
        f"--runtime-env-json={json.dumps(OmegaConf.to_container(cfg.gpu.ray_job.runtime_env, resolve=True), separators=(',', ':'))}",
        "--",
        str(cfg.runtime.python_executable),
        str(cfg.runtime.train_entrypoint),
        *train_args,
    ]
    _run_command(submit_cmd, env=env)


if __name__ == "__main__":
    main()
