"""
MathEnvClient — local (serverless) environment for math problem solving.

Constructed per-problem with (problem, label). Uses math_verify.parse/verify
for answer checking.
"""
import re

from math_verify import parse, verify
from slime.rollout.rm_hub.math_utils import grade_answer_verl
from slime.rollout.rm_hub.math_utils import last_boxed_only_string

from .controller.types import ParseResult, StepOutput


class MathEnvClient:
    def __init__(self, problem: str, label: str):
        self.problem = problem
        self.label = label

    def observe(self) -> str:
        return self.problem

    def parse_response(self, response: str) -> ParseResult:
        boxed_answer = last_boxed_only_string(response)
        if boxed_answer is not None:
            return ParseResult(type="action", content=boxed_answer.strip())
        return ParseResult(type=None, content=response)

    def step(self, action: str) -> StepOutput:
        candidate = action.strip()
        gold = parse(str(self.label), parsing_timeout=None)
        pred = parse(candidate, parsing_timeout=None)
        correct = bool(pred) and bool(verify(gold or str(self.label), pred, strict=True, timeout_seconds=None))
        if not correct:
            correct = grade_answer_verl(candidate, self.label)
            if not correct and "\\boxed" not in candidate:
                correct = grade_answer_verl(f"\\boxed{{{candidate}}}", self.label)
        reward = 1.0 if correct else 0.0
        state = "Your answer is correct." if correct else "Your answer is incorrect."
        if not pred and not correct:
            state = "Could not extract a valid final answer from your response."
        return StepOutput(
            state=state,
            reward=reward,
            done=True,
        )

    @property
    def invalid_action_obs(self) -> str:
        return "Invalid format. Put your final answer in \\boxed{your_answer}."

    def close(self):
        pass
