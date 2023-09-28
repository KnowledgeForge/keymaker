"""Constraints for stop patterns"""
from dataclasses import dataclass

import regex as re

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint


@dataclass
class StopsConstraint(Constraint):
    """Stop generation on occurence of a fixed string

    Attributes:
        stop (str): The string or regex pattern that when hit stops completion.
        include (bool): Whether to include the stop string or pattern in the completion or not.
    """

    stop: str
    include: bool = True

    def __post_init__(self):
        self._pattern = re.compile(rf"(?P<completion>.*?)(?P<stop>{self.stop})")

    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: Model,
    ) -> TokenConstraint:
        match = self._pattern.search(completion_text)
        if match:
            if not self.include:
                return completion_text
            return completion_text + match.group("stop")
        return None
