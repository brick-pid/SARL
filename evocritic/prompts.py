from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2 import meta

WEBSHOP_DESC = """
Webshop is an e-commerce shopping environment where you need to find and purchase product based on given shopping goal.
""".strip()

WEBSHOP_ACTIONS = """
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.

There are different types of pages:
- Initial Page: You can perform search actions to find products.
- Search Results Page: You can view search results, navigate through pages using click[Next >] and click[< Prev], and click on product ASIN to view details.
- Product Detail Page: You can view product details, check description and features, and if the product matches the requirements, you can purchase the product. If not, you can go back to the search results or go to initial page.

Available Actions:
- <action>search[query]</action>: search for products using the specified query (initial page only).
- <action>click[button]</action>: navigate by clicking buttons or items.
""".strip()

SCIWORLD_DESC = """
SciWorld is a scientific interactive environment with rooms, containers, tools, substances, and devices.
You need to complete a science task by exploring, manipulating objects, and performing valid action sequences.
""".strip()

SCIWORLD_ACTIONS = """
Available Actions:
- <action>open [OBJ]</action>: open a container.
- <action>close [OBJ]</action>: close a container.
- <action>activate [OBJ]</action>: activate a device (e.g. turn on a stove to heat something).
- <action>deactivate [OBJ]</action>: deactivate a device.
- <action>connect [OBJ] to [OBJ]</action>: connect electrical components to create a circuit.
- <action>disconnect [OBJ]</action>: disconnect electrical components.
- <action>use [OBJ] [on OBJ]</action>: use a device/item as a tool, optionally on another object.
- <action>look around</action>: describe the current room.
- <action>look at [OBJ]</action>: describe an object in detail.
- <action>look in [OBJ]</action>: describe a container's contents.
- <action>read [OBJ]</action>: read a note or book.
- <action>move [OBJ] to [OBJ]</action>: move an object to a container.
- <action>pick up [OBJ]</action>: move an object to the inventory.
- <action>put down [OBJ]</action>: drop an inventory item.
- <action>pour [OBJ] into [OBJ]</action>: pour a liquid into a container.
- <action>dunk [OBJ] into [OBJ]</action>: dunk a container into a liquid.
- <action>mix [OBJ]</action>: chemically mix the contents of a container.
- <action>go to [LOC]</action>: move to a new location.
- <action>eat [OBJ]</action>: eat a food.
- <action>flush [OBJ]</action>: flush a toilet.
- <action>focus on [OBJ]</action>: signal intent on a task object.
- <action>wait</action>: take no action for 10 iterations.
- <action>wait1</action>: take no action for 1 iteration.
- <action>task</action>: describe the current task.
- <action>inventory</action>: list your inventory.
""".strip()

ALFWORLD_DESC = """
ALFWorld is a household environment with rooms, receptacles, and manipulable objects.
You need to solve a household goal by navigating and interacting with objects and containers.
""".strip()

ALFWORLD_ACTIONS = """
Available Actions:
- <action>go to [LOCATION]</action>: move to a receptacle or location.
- <action>take [OBJECT] from [RECEPTACLE]</action>: pick up an object from a receptacle.
- <action>put [OBJECT] in/on [RECEPTACLE]</action>: put an object on a receptacle.
- <action>open [RECEPTACLE]</action>: open a receptacle to reveal its contents.
- <action>close [RECEPTACLE]</action>: close a receptacle.
- <action>toggle [OBJECT] [RECEPTACLE]</action>: switch an object on or off.
- <action>clean [OBJECT] with [RECEPTACLE]</action>: clean an object using a receptacle.
- <action>heat [OBJECT] with [RECEPTACLE]</action>: heat an object using a receptacle.
- <action>cool [OBJECT] with [RECEPTACLE]</action>: cool an object using a receptacle.
- <action>inventory</action>: list objects currently being carried.
- <action>look</action>: describe the current situation.
- <action>examine [OBJECT]</action>: examine an object or receptacle in detail.
""".strip()

SEARCHQA_DESC = """
SearchEnv is a websearch environment. You need to answer a question by issuing search queries and iteratively refining your search based on retrieved information.
When all necessary information is gathered, return a short and concise final answer.
""".strip()

SEARCHQA_ACTIONS = """
Available Actions:
- <search>query</search>: search for relevant information.
- <answer>answer</answer>: provide the final concise answer.

When giving the final answer, make it short and concise. Don't include any additional explanations or notes.
For example, if the question is "What is the capital of France?":
<answer>Paris</answer>
""".strip()

WEBARENA_DESC = """
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.
""".strip()

WEBARENA_ACTIONS = """
The actions you can perform fall into several categories:

Page Operation Actions:
- <action>click [id]</action>: click on an element with a specific id on the webpage.
- <action>type [id] [content] [0|1]</action>: type the content into the field with id. By default, the "Enter" key is pressed after typing unless the last parameter is set to 0.
- <action>hover [id]</action>: hover over an element with id.
- <action>press [key_comb]</action>: simulate the pressing of a key combination on the keyboard (e.g., Ctrl+v).
- <action>scroll [down|up]</action>: scroll the page up or down to reveal content below or above the current view.

Tab Management Actions:
- <action>new_tab</action>: open a new, empty browser tab.
- <action>tab_focus [tab_index]</action>: switch the browser's focus to a specific tab using its index.
- <action>close_tab</action>: close the currently active tab.

URL Navigation Actions:
- <action>goto [url]</action>: navigate to a specific URL.
- <action>go_back</action>: navigate to the previously viewed page.
- <action>go_forward</action>: navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
- <action>stop [answer]</action>: issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.

Homepage:
If you want to visit other websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.

Rules:
1. You should only issue an action that is valid given the current observation.
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. For ALL actions that take parameters, you MUST enclose each parameter in square brackets [].
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.
""".strip()

MATH_DESC = """
Math is a single-turn problem-solving environment. You are given one math problem and must return only the final answer in the required XML answer tag.
""".strip()

MATH_ACTIONS = """
Available Actions:
<answer>
\\boxed{answer}
</answer>
""".strip()

ENV_REGISTRY = {
    "webshop": {"desc": WEBSHOP_DESC, "actions": WEBSHOP_ACTIONS},
    "sciworld": {"desc": SCIWORLD_DESC, "actions": SCIWORLD_ACTIONS},
    "alfworld": {"desc": ALFWORLD_DESC, "actions": ALFWORLD_ACTIONS},
    "searchqa": {"desc": SEARCHQA_DESC, "actions": SEARCHQA_ACTIONS},
    "webarena": {"desc": WEBARENA_DESC, "actions": WEBARENA_ACTIONS},
    "math": {"desc": MATH_DESC, "actions": MATH_ACTIONS},
}

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
