"""
MathEnvClient — local (serverless) environment for math problem solving.

Constructed per-problem with (problem, label). Uses math_verify.parse/verify
for answer checking.
"""
import re

from math_verify import parse, verify

from .controller.types import ParseResult, StepOutput


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
        candidate = action.strip()
        gold = parse(str(self.label), parsing_timeout=None)
        pred = parse(candidate, parsing_timeout=None)
        correct = bool(pred) and bool(verify(gold or str(self.label), pred, strict=True, timeout_seconds=None))
        reward = 1.0 if correct else 0.0
        state = "Your answer is correct." if correct else "Your answer is incorrect."
        if not pred:
            state = "Could not extract a valid final answer from your response."
        return StepOutput(
            state=state,
            reward=reward,
            done=True,
        )

    @property
    def invalid_action_obs(self) -> str:
        return "Invalid format. Use <answer>\\boxed{your_answer}</answer>."

    def close(self):
        pass
