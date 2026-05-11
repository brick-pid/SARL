from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from .controller import BaseEnvClient, BaseTask
from .controller.types import StepOutput


class WebarenaEnvClient(BaseEnvClient):
    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()["env_idx"]

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
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action has already been parsed by tool_parser (e.g. "click [id]").
        # Wrap in backticks so the server-side extract_action can parse it.
        wrapped = f"```{action}```"
        response = self._post("step", {"action": wrapped})
        reward = response["reward"] if response["terminated"] else 0
        return StepOutput(
            state=response["observation"],
            reward=reward,
            done=response["terminated"],
        )

    def reset(self, idx: int) -> Dict[str, Any]:
        response = self._post("reset", {"seed": 0, "idx": idx})
        if response["observation"] == "TimeoutError":
            raise TimeoutError(f"WebArena Reset Timeout: item id={idx}, you may consider restarting the web server.")
        return response

    def close(self):
        response = self._post("close",{})
        return response

class WebarenaTask(BaseTask):
    env_client_cls = WebarenaEnvClient
    env_name = "Webarena"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
