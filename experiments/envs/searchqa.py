import re
from typing import Any, Mapping, Dict, List, Optional

import requests
from requests.exceptions import RequestException
from .controller import BaseEnvClient, BaseTask
from .controller.types import ConversationMessage, StepOutput

def parse_action(action_str: str) -> tuple[str | None, str | None]:
    # parse format of "act[string]", return act and string
    pattern = r"(\w+)\[(.*)\]"
    matches = re.findall(pattern, action_str)
    if not matches:
        return None, None
    act, string = matches[-1]
    return act, string

class SearchQAEnvClient(BaseEnvClient):
    conversation_start = (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value":"""
You should first reason about how to solve the question, then think carefully which search query best advances answering the question. Once you've finished your reasoning, you should choose a search query for current step and present it within <action> </action> tags. Available Actions:\n- <action> search[query] </action>: search for relevant information.\n- <action> answer[answer] </action>: provide the final concise answer. When giving the final answer, make it short and concise. Don't include any additional explanations or notes. For example, if the question is "What is the capital of France?" and you have found the answer to be "Paris", you should respond with: <action> answer[Paris] </action>""".strip(),
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data['id'] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        question = self._get("observation")
        return question

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        # print(f"Action: {action}")
        act, content = parse_action(action)
        if act == "answer":
            action = f"<answer> {content} </answer>"
        elif act == "search":
            action = f"<search> {content} </search>"
        else:
            state = "Invalid action format. Please use <action>search[query]</action> or <action>answer[answer]</action>."
            return StepOutput(state=state, reward=0.0, done=False)
        response = self._post("step", {"action": action})
        # print(response)
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response
    
    def close(self):
        response = self._post("close", {})
        return response

class SearchQATask(BaseTask):
    env_client_cls = SearchQAEnvClient
    env_name = "SearchQA"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
