from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from .controller import (
    BaseEnvClient,
    BaseTask,
)
from .controller.types import StepOutput


class SciworldEnvClient(BaseEnvClient):

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, action_format = "react_xml", **kwargs
    ):
        super().__init__(action_format=action_format, *args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        ok = ok.json()
        self.env_id = ok["id"]

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        response = self._post("step", {"action": action})
        if "No known action matches" in response["observation"]:
            response["observation"] = self.observe()
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["task_description"] + '\n' + response["observation"],
            "reward": 0,
            "score": 0,
            "done": False,
        }
        return response

    def close(self):
        response = self._post("close",{})
        return response

class SciworldTask(BaseTask):
    env_client_cls = SciworldEnvClient
    env_name = "SciWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
