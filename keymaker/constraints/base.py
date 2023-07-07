"""Base Constraint interface"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from keymaker.models.base import Model

from keymaker.types import TokenConstraint


class Constraint(ABC):
    @abstractmethod
    def constrain_tokens(self, base_text: str, completion_text: str, model: "Model") -> TokenConstraint:
        """Constrain the token ids that can be sampled from the model's vocabulary.

        Args:
            base_text (str): The text to which the completion_text should be appended.
            completion_text (str): The text to be completed.
            model (Model): The language model to be used.

        Returns:
            None: If no restrictions are to be applied and the full vocabulary can be used.
            set: The set of valid token ids that can be sampled.
            str: If the constraint is complete and the str is the finished value, which may not be what was passed as the completion text.
        """

    def __or__(self, other):
        from keymaker.constraints.logical import OrConstraint

        return OrConstraint([self, other])

    def __and__(self, other):
        from keymaker.constraints.logical import AndConstraint

        return AndConstraint([self, other])
