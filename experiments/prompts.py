from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2 import meta

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

WEBSHOP_ENV_DESC = """
Webshop is a e-commerce shopping environment where you need to find and purchase product based on given shoping goal. \
You should first reason step-by-step about the current situation, then think carefully which available action best advances the shopping goal. 
Once you've finished your reasoning, you should choose an available action for current step and present it within <action> </action> tags.
""".strip()

WEBSHOP_ENV_ACTION = """
Every round I will give you an observation and a list of available actions, you have to respond to an action based on the state and instruction. \
You can use search action if the search is available. You can click one of the buttons in clickables. If the action is not valid, perform nothing. \
Keywords in search are up to you, but the value in click must be a value in the list of available actions. Remember that your keywords in search should be carefully designed.

There are different types of pages:
- Initial Page: You can perform search actions to find products.
- Search Results Page: You can view search results, navigate through pages using click[Next >] and click[< Prev], and click on product ASIN to view details.
- Product Detail Page: You can view product details, check description and features, and if the product matches the requirements, you can purchase the product. If not, you can go back to the search results or go to initial page.

Available Actions:
- <action>search[query]</action>: search for products using the specified query (initial page only).
- <action>click[button]</action>: navigate by clicking buttons or items.
""".strip()

SCIWORLD_ENV_DESC = """
SciWorld is a scientific interactive environment with rooms, containers, tools, substances, and devices.
You need to complete a science task by exploring, manipulating objects, and performing valid action sequences.
""".strip()

SCIWORLD_ENV_ACTION = """
You should first reason step-by-step about the current situation, then think carefully which available action best advances the research goal. 
Once you've finished your reasoning, you should choose an available action for current step and present it within <action> </action> tags.

Available Actions:
- <action>open [OBJ]</action>
- <action>close [OBJ]</action>
- <action>activate [OBJ]</action>
- <action>deactivate [OBJ]</action>
- <action>connect [OBJ] to [OBJ]</action>
- <action>disconnect [OBJ]</action>
- <action>use [OBJ] [on OBJ]</action>
- <action>look around</action>
- <action>look at [OBJ]</action>
- <action>look in [OBJ]</action>
- <action>read [OBJ]</action>
- <action>move [OBJ] to [OBJ]</action>
- <action>pick up [OBJ]</action>
- <action>put down [OBJ]</action>
- <action>pour [OBJ] into [OBJ]</action>
- <action>dunk [OBJ] into [OBJ]</action>
- <action>mix [OBJ]</action>
- <action>go to [LOC]</action>
- <action>eat [OBJ]</action>
- <action>flush [OBJ]</action>
- <action>focus on [OBJ]</action>
- <action>wait</action>
- <action>wait1</action>
- <action>task</action>
- <action>inventory</action>
""".strip()

ALFWORLD_ENV_DESC = """
ALFWorld is a household environment with rooms, receptacles, and manipulable objects.
You need to solve a household goal by navigating and interacting with objects and containers.
""".strip()

ALFWORLD_ENV_ACTION = """
You should first reason step-by-step about the current situation, then think carefully which available action best advances the household task. 
Once you've finished your reasoning, you should choose an available action for current step and present it within <action> </action> tags.

Available Actions:
- <action>go to [LOCATION]</action>
- <action>take [OBJECT] from [RECEPTACLE]</action>
- <action>put [OBJECT] in/on [RECEPTACLE]</action>
- <action>open [RECEPTACLE]</action>
- <action>close [RECEPTACLE]</action>
- <action>toggle [OBJECT] [RECEPTACLE]</action>
- <action>clean [OBJECT] with [RECEPTACLE]</action>
- <action>heat [OBJECT] with [RECEPTACLE]</action>
- <action>cool [OBJECT] with [RECEPTACLE]</action>
- <action>inventory</action>
- <action>look</action>
- <action>examine [OBJECT]</action>
""".strip()

SEARCHQA_ENV_DESC = """
SearchEnv is a websearch environment. You need to answer a question by issuing search queries and iteratively refining your search based on retrieved information.
When all necessary information is gathered, return a short and concise final answer.
""".strip()

SEARCHQA_ENV_ACTION = """
You should first reason step-by-step about the current situation, then think carefully which search query best advances answering the question.
Once you've finished your reasoning, you should choose a search query for current step and present it within <action> </action> tags.

Available Actions:
- <action>search[query]</action>: search for relevant information.
- <action>answer[answer]</action>: provide the final concise answer.

When giving the final answer, make it short and concise. Don't include any additional explanations or notes.
For example, if the question is "What is the capital of France?" and you have found the answer to be "Paris", you should respond with:
<action>answer[Paris]</action>
""".strip()

ENV_DESC = {
    "webshop": WEBSHOP_ENV_DESC,
    "sciworld": SCIWORLD_ENV_DESC,
    "alfworld": ALFWORLD_ENV_DESC,
    "searchqa": SEARCHQA_ENV_DESC,
}

def _validate_template_vars(template_name: str, context: dict[str, Any]) -> None:
    source, _, _ = _TEMPLATE_ENV.loader.get_source(_TEMPLATE_ENV, template_name)
    ast = _TEMPLATE_ENV.parse(source)
    required_vars = meta.find_undeclared_variables(ast)
    missing_vars = sorted(
        var for var in required_vars if var not in context or context[var] is None
    )
    if missing_vars:
        raise ValueError(
            f"Missing required template variables for {template_name}: {', '.join(missing_vars)}"
        )

def render_main_system_prompt(env_name: str, task: str) -> str:
    env_desc = ENV_DESC[env_name]
    template = _TEMPLATE_ENV.get_template("main_system.j2")
    context = {
        "env_name": env_name,
        "env_desc": env_desc,
        "task": task,
    }
    _validate_template_vars("main_system.j2", context)
    return template.render(**context).strip()


def render_subagent_system_prompt(env_name: str, subtask: str, env_inst) -> str:
    template = _TEMPLATE_ENV.get_template("subagent_system.j2")
    context = {
        "env_name": env_name,
        "env_inst": env_inst,
        "subtask": subtask,
    }
    _validate_template_vars("subagent_system.j2", context)
    return template.render(**context).strip()
