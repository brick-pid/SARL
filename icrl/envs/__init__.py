from .academia import AcademiaEnvClient
from .alfworld import AlfWorldEnvClient
from .babyai import BabyAIEnvClient
from .lmrlgym import MazeEnvClient, WordleEnvClient
from .movie import MovieEnvClient
from .sciworld import SciworldEnvClient
from .sheet import SheetEnvClient
from .sqlgym import SqlGymEnvClient
from .textcraft import TextCraftEnvClient
from .todo import TodoEnvClient
from .weather import WeatherEnvClient
from .webarena import WebarenaEnvClient
from .webshop import WebshopEnvClient
from .searchqa import SearchQAEnvClient
from .math_env import MathEnvClient

import time


def init_env_client(env_name, env_addr, max_retries=3, action_format="react_xml"):
    envclient_classes = {
        "webshop": WebshopEnvClient,
        "alfworld": AlfWorldEnvClient,
        "babyai": BabyAIEnvClient,
        "sciworld": SciworldEnvClient,
        "textcraft": TextCraftEnvClient,
        "webarena": WebarenaEnvClient,
        "sqlgym": SqlGymEnvClient,
        "maze": MazeEnvClient,
        "wordle": WordleEnvClient,
        "weather": WeatherEnvClient,
        "todo": TodoEnvClient,
        "movie": MovieEnvClient,
        "sheet": SheetEnvClient,
        "academia": AcademiaEnvClient,
        "searchqa": SearchQAEnvClient,
    }
    envclient_class = envclient_classes.get(env_name)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {env_name}")

    retry = 0
    while True:
        try:
            return envclient_class(
                env_server_base=env_addr,
                data_len=1,
                timeout=2400,
                action_format=action_format,
            )
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server {env_addr}, retrying...({retry}/{max_retries})")
            if retry > max_retries:
                raise e
            time.sleep(5)
