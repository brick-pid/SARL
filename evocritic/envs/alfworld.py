from typing import Any, Mapping

import requests

from .controller import BaseEnvClient, BaseTask
from .controller.types import StepOutput

class AlfWorldEnvClient(BaseEnvClient):
    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        
        ok = ok.json()
        # print(ok)
        self.env_id = ok["id"]
        self.info = None

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
        return f"{self.info['observation']}\nAVAILABLE ACTIONS: {','.join(self.info['available_actions'])}"

    def step(self, action: str) -> StepOutput:
        # print(f"Action: {action}")
        response = self._post("step", {"action": action})
        # print(response)
        self.info = {
            "observation": response["observation"],
            "available_actions": response["available_actions"],
            "reward": response["reward"],
            "done": response["done"],
        }
        if "Nothing happens" in response["observation"]:
            response["observation"] = self.observe()
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, game: int, world_type: str = "Text") -> dict[str, Any]:
        response = self._post("reset", {"game": game, "world_type": world_type})
        self.info = {
            "observation": response["observation"],
            "available_actions": response["available_actions"],
            "reward": 0,
            "done": False,
        }
        return response

    def close(self):
        response = self._post("close",{})
        return response

class AlfWorldTask(BaseTask):
    env_client_cls = AlfWorldEnvClient
    env_name = "AlfWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
