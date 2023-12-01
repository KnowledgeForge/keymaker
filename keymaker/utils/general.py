"""General keymaker utils"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from keymaker.prompt import CompletionConfig


noop = lambda x: x  # noqa: E731


async def anoop(x):
    """A noop coroutine function."""
    return x


def add_logprob(logprobs_sum: Optional[float], *logprobs: Optional[float]) -> Optional[float]:
    """Adds together log probabilities.

    Args:
        logprobs_sum: The sum of the log probabilities already calculated or None.
        *logprobs: The log probabilities to add to the sum.

    Returns:
        The updated sum of log probabilities if valid, otherwise None.
    """
    if logprobs_sum is None or None in logprobs:
        return None
    return logprobs_sum + sum(logprobs)  # type: ignore


def exp(x: Optional[float]) -> Optional[float]:
    """Computes the exponential function of x.

    Args:
        x: The number to compute the exponential of.

    Returns:
        The exponential of x if x is not None, otherwise None.
    """
    if x is None:
        return None
    return math.exp(x)


@dataclass
class TokenTracker:
    """Tracks token counts."""

    _counts: List["TokenCount"] = field(default_factory=list)

    def add_token_count(self, count: "TokenCount"):
        """Adds a token count.

        Args:
            count: The token count to add.
        """
        self._counts.append(count)

    @property
    def counts(self) -> List["TokenCount"]:
        """Gets the tracked token counts.

        Returns:
            The list of tracked token counts.
        """
        return self._counts


@dataclass(eq=True)
class TokenCount:
    """
    A class for counting prompt and completion tokens.

    Properties:
        model (Optional[Model]): The model that generated the tokens.
        completion (Optional[CompletionConfig]): The completion configuration used to generate the tokens.
        prompt_tokens (int): The count of prompt tokens.
        completion_tokens (int): The count of completion tokens.
        tokens (int): The total count of tokens (sum of prompt and completion tokens).
    """

    _completion_config: Optional["CompletionConfig"] = None
    _prompt_tokens: int = 0
    _completion_tokens: int = 0

    def __repr__(self) -> str:
        return f"TokenCount(completion_config = {self._completion_config}, prompt_tokens = {self._prompt_tokens}, completion_tokens = {self._completion_tokens})"

    def set_config(self, config: "CompletionConfig"):
        if self._completion_config is not None:
            raise ValueError(f"`completion_config` already set on {self}.")
        self._completion_config = config

    def add_prompt_tokens(self, count: int = 1):
        """
        Increment the count of prompt tokens by the specified amount.

        Args:
            count (int): The number of prompt tokens to increment. Defaults to 1.
        """
        self._prompt_tokens += count

    def add_completion_tokens(self, count: int = 1):
        """
        Increment the count of completion tokens by the specified amount.

        Args:
            count (int): The number of completion tokens to increment. Defaults to 1.
        """
        self._completion_tokens += count

    @property
    def completion_config(self) -> Optional["CompletionConfig"]:
        """
        Get the complet

        Returns:
            int: The count of prompt tokens.
        """
        return self._completion_config

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
