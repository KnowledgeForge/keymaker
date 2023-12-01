"""Utils for facilitating completing format strings"""

import re
from dataclasses import dataclass
from typing import Optional

from parsy import generate, regex, string


@dataclass
class Format:
    name: Optional[str]


lbrack = string("{")
rbrack = string("}")


def format_parser(s: str):
    chunks = []

    @generate
    def helper():
        text = yield regex(r"[^{}]*")
        bracket1 = yield lbrack.optional()
        bracket2 = yield lbrack.optional()
        if bracket1 and bracket2:
            rest = yield (regex(r".*}}", flags=re.DOTALL))
            return [text + "{"] + format_parser(rest[:-2]) + ["}"]
        if bracket1:
            inner = yield regex(r"\s*[a-zA-Z_]*[a-zA-Z0-9_]*\s*") | regex(r"\s*[a-zA-Z_][a-zA-Z0-9_]*\s*")
            yield rbrack
            return [text, Format(inner or None)]
        return [text]

    while s:
        chunk, s = helper.parse_partial(s)
        chunks += chunk
    return chunks
