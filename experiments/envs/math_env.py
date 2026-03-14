"""
MathEnvClient — local (serverless) environment for math problem solving.

Constructed per-problem with (problem, label). Uses verify() from
slime/rollout/rm_hub/math_dapo_utils.py for answer checking.
"""
import re

from .controller.types import ParseResult, StepOutput
from slime.rollout.rm_hub.math_dapo_utils import verify


class MathEnvClient:
    def __init__(self, problem: str, label: str):
        self.problem = problem
        self.label = label

    def observe(self) -> str:
        return self.problem

    def parse_response(self, response: str) -> ParseResult:
        # Check for <subagent>...</subagent>
        sub_match = list(re.finditer(r"<subagent>(.*?)</subagent>", response, re.DOTALL))
        if sub_match:
            return ParseResult(type="subagent", content=sub_match[-1].group(1).strip())

        # Check for <answer>...</answer>
        ans_match = list(re.finditer(r"<answer>(.*?)</answer>", response, re.DOTALL))
        if ans_match:
            return ParseResult(type="action", content=ans_match[-1].group(1).strip())

        return ParseResult(type=None, content=response)

    def step(self, action: str) -> StepOutput:
        correct, pred = verify(solution_str=action, answer=self.label)
        reward = 1.0 if correct else 0.0
        return StepOutput(
            state=f"Your answer is {'correct' if correct else 'incorrect'}.",
            reward=reward,
            done=True,
        )

    @property
    def invalid_action_obs(self) -> str:
        return "Invalid format. Use <answer>your_answer</answer>."

    def close(self):
        pass
