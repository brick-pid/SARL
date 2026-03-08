import re
from abc import ABCMeta, abstractmethod

from .types import ActionFormat, ConversationMessage, ParseResult, StepOutput


class BaseEnvClient(metaclass=ABCMeta):
    _conversation_start: dict[ActionFormat, tuple[ConversationMessage]]

    def __init__(self, action_format: ActionFormat = "react") -> None:
        self.action_format = ActionFormat(action_format)

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total size of the environment.
        """

    @abstractmethod
    def observe(self) -> str:
        """
        Parse env server response and give a text message to prompt the LLM.
        """

    @abstractmethod
    def step(self, action) -> StepOutput:
        """
        Parse model output from the action and call the env server.
        """

    @abstractmethod
    def reset(self, idx: int) -> None:
        """
        Reset the environment.
        """

    def parse_response(self, response: str) -> ParseResult:
        """Parse LLM response. Override for custom action formats."""
        pattern = r"<(action|subagent)>(.*?)</\1>"
        matches = list(re.finditer(pattern, response, re.DOTALL))
        if not matches:
            return ParseResult(type=None, content=response)
        m = matches[-1]
        return ParseResult(type=m.group(1), content=m.group(2).strip())

    @property
    def invalid_action_obs(self) -> str:
        return ("The task is not completed yet. Think more carefully about how to "
                "interact with the environment to complete the task. "
                "Response should include <action> your action </action>.")
