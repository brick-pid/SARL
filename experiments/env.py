import logging

import httpx
import requests

logger = logging.getLogger(__name__)


class GymEnv:
    """
    OpenAI Gym-style environment interface.

    Communicates with a remote environment server via HTTP POST requests.
    Supports: webshop, alfworld, sciworld, searchqa.

    Server API:
        POST /create  -> {"env_id": str}, 这个方法只被内部 init 调用
        POST /reset   -> {"observation": str, "info": dict}
        POST /step    -> {"observation": str, "reward": float, "done": bool, "info": dict}
        POST /close   -> {"closed": bool, "env_id": str}
    """

    def __init__(self, env_name: str, address: str):
        assert env_name in ["webshop", "alfworld", "sciworld", "searchqa"]
        self.env_name = env_name
        self.address = address.rstrip("/")
        self.env_id = self._create()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    def _create(self) -> str:
        response = requests.post(f"{self.address}/create")
        response.raise_for_status()
        data = response.json()
        return data["env_id"]

    async def reset(self, task_id: int) -> tuple[str, dict]:
        self.task_id = task_id
        payload = {"env_id": self.env_id, "task_id": self.task_id}
        response = await self._client.post(f"{self.address}/reset", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["observation"], data["info"]

    async def step(self, action: str) -> tuple[str, float, bool, dict]:
        payload = {"env_id": self.env_id, "action": action}
        response = await self._client.post(f"{self.address}/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["observation"], data["reward"], data["done"], data["info"]

    async def close(self) -> None:
        payload = {"env_id": self.env_id}
        response = await self._client.post(f"{self.address}/close", json=payload)
        response.raise_for_status()
        await self._client.aclose()

if __name__ == "__main__":
    import asyncio

    async def main():
        env = GymEnv("alfworld", "http://127.0.0.1:36002")
        print(env.env_id)
        obs, info = await env.reset(task_id=0)
        print(f"obs: {obs}\n\n info: {info}")
        obs, reward, done, info = await env.step(action="go to drawer 2")
        print(obs, reward, done, info)
        await env.close()

    asyncio.run(main())
    
    