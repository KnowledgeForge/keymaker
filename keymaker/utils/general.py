"""General keymaker utils"""

from typing import Optional
import math

noop = lambda x: x


def add_logprob(logprobs_sum: Optional[float], *logprobs: Optional[float])->Optional[float]:
    if logprobs_sum is None or None in logprobs:
        return None
    return logprobs_sum + sum(logprobs)

def exp(x: Optional[float])->Optional[float]:
    if x is None:
        return None
    return math.exp(x)