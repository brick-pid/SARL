"""
A simple tool parser

support format:
<action> act_str </action>
<subagent> act_str </subagent>
"""
import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    type: str # in ("action", "subagent")
    content: str # action/subagent input/conclusion

def tool_parser(response: str) -> ParseResult | None:
    pattern = r"<(action|subagent|conclusion)>(.*?)</\1>"
    matches = list(re.finditer(pattern, response, re.DOTALL))
    if not matches:
        return None
    m = matches[-1]
    return ParseResult(type=m.group(1), content=m.group(2).strip())