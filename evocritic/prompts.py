from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2 import meta

from experiments.prompts import ENV_REGISTRY

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

_ROLE_TEMPLATE = {
    "executor": "executor_system.j2",
    "verifier": "verifier_system.j2",
    "critic": "critic_system.j2",
}

_MATH_ROLE_TEMPLATE = {
    "executor": "math_executor_system.j2",
    "critic": "math_critic_system.j2",
}


def _validate_template_vars(template_name: str, context: dict[str, Any]) -> None:
    source, _, _ = _TEMPLATE_ENV.loader.get_source(_TEMPLATE_ENV, template_name)
    ast = _TEMPLATE_ENV.parse(source)
    required_vars = meta.find_undeclared_variables(ast)
    missing_vars = sorted(var for var in required_vars if var not in context or context[var] is None)
    if missing_vars:
        raise ValueError(
            f"Missing required template variables for {template_name}: {', '.join(missing_vars)}"
        )


def render_role_prompt(
    *,
    env_name: str,
    role: str,
    task: str,
) -> str:
    if role not in _ROLE_TEMPLATE:
        raise ValueError(f"Unknown role: {role!r}. Expected one of {sorted(_ROLE_TEMPLATE)}")
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_name: {env_name!r}")

    reg = ENV_REGISTRY[env_name]
    template_map = _MATH_ROLE_TEMPLATE if env_name == "math" else _ROLE_TEMPLATE
    template_name = template_map[role]
    template = _TEMPLATE_ENV.get_template(template_name)
    context = {
        "env_name": env_name,
        "env_desc": reg["desc"],
        "env_actions": reg["actions"],
        "task": task,
    }
    _validate_template_vars(template_name, context)
    return template.render(**context).strip()
