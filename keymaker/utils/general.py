"""General keymaker utils"""

import math
from typing import Optional
from dataclasses import dataclass

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

from dataclasses import dataclass

@dataclass(eq=True)
class TokenCounter:
    """
    A class for counting prompt and completion tokens.

    Properties:
        prompt_tokens (int): The count of prompt tokens.
        completion_tokens (int): The count of completion tokens.
        tokens (int): The total count of tokens (sum of prompt and completion tokens).
    """

    _prompt_tokens: int = 0
    _completion_tokens: int = 0

    def __repr__(self)->str:
        return f"TokenCounter(prompt_tokens={self._prompt_tokens}, completion_tokens={self._completion_tokens})"

    def _prompt(self, count: int = 1):
        """
        Increment the count of prompt tokens by the specified amount.

        Args:
            count (int): The number of prompt tokens to increment. Defaults to 1.
        """
        self._prompt_tokens += count

    def _completion(self, count: int = 1):
        """
        Increment the count of completion tokens by the specified amount.

        Args:
            count (int): The number of completion tokens to increment. Defaults to 1.
        """
        self._completion_tokens += count

    @property
    def prompt_tokens(self) -> int:
        """
        Get the count of prompt tokens.

        Returns:
            int: The count of prompt tokens.
        """
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """
        Get the count of completion tokens.

        Returns:
            int: The count of completion tokens.
        """
        return self._completion_tokens

    @property
    def tokens(self) -> int:
        """
        Get the total count of tokens (sum of prompt and completion tokens).

        Returns:
            int: The total count of tokens.
        """
        return self.prompt_tokens + self.completion_tokens


