import re
from typing import Any, Mapping, Dict, List, Optional

import requests
from requests.exceptions import RequestException
from .controller import BaseEnvClient, BaseTask
from .controller.types import ParseResult, StepOutput


class SearchQAEnvClient(BaseEnvClient):
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

    def parse_response(self, response: str) -> ParseResult:
        """Match SearchQA actions or main-agent subagent delegation tags."""
        pattern = r"<(search|answer|action|subagent)>(.*?)</\1>"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            return ParseResult(type=None, content=response)
        m = matches[-1]
        tag = m.group(1)
        inner = m.group(2).strip()
        if tag == "subagent":
            return ParseResult(type="subagent", content=inner)
        return ParseResult(type="action", content=f"<{tag}>{inner}</{tag}>")

    @property
    def invalid_action_obs(self) -> str:
        return ("The task is not completed yet. Think more carefully about how to "
                "interact with the environment to complete the task. "
                "Response should include <search>query</search> or <answer>answer</answer>.")

    def step(self, action: str) -> StepOutput:
        response = self._post("step", {"action": action})
        is_answer_action = bool(re.search(r"<answer>.*?</answer>", action, re.DOTALL))
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"] or is_answer_action,
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
