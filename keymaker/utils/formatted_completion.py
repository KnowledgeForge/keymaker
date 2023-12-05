"""Utils for facilitating completing format strings"""

from dataclasses import dataclass
from string import Formatter
from typing import Optional


@dataclass
class Format:
    name: Optional[str]


def format_parser(s: str):
    for pre, fname, post, *_ in Formatter().parse(s):
        yield pre
        if fname is not None:
            yield Format(fname)
        if post is not None:
            yield post
