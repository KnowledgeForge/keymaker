"""General keymaker utils"""

import math
from typing import Optional

noop = lambda x: x  # noqa: E731


async def anoop(x):
    return x


def add_logprob(logprobs_sum: Optional[float], *logprobs: Optional[float]) -> Optional[float]:
    if logprobs_sum is None or None in logprobs:
        return None
    return logprobs_sum + sum(logprobs)  # type: ignore


def exp(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return math.exp(x)
