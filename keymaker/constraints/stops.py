"""Constraints for regex patterns"""
from dataclasses import dataclass

from keymaker.constraints.base import Constraint
from keymaker.constraints.regex import RegexConstraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint


@dataclass
class StopsConstraint(Constraint):
    """Stop generation on occurence of a fixed string

    Attributes:
        stop (str): The string after which to stop.
        include (bool): Whether to include the stop string in the completion or not.
    """

    stop: str
    include: bool = True

    def __post_init__(self):
        end = self.stop
        if not self.include:
            end = f"(?={self.stop})"
        self._re_constraint = RegexConstraint(".*?" + end)

    def constrain_tokens(self, base_text: str, completion_text: str, model: Model) -> TokenConstraint:
        return self._re_constraint.constrain_tokens(base_text, completion_text, model)
