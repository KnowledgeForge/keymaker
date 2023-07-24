"""Base Constraint interface"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from keymaker.models.base import Model

from keymaker.types import TokenConstraint


class Constraint(ABC):
    @abstractmethod
    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: "Model",
    ) -> TokenConstraint:
        """Constrain the token ids that can be sampled from the model's vocabulary.

            Args:
                base_text (str): The text to which the completion_text should be appended.
                completion_text (str): The text completed thus far; one token will be added between calls
                model (Model): The language model to be used.

        Returns:
            A two-tuple:
                - One of
                    - None: If no restrictions are to be applied and the full vocabulary can be used.
                    - Set[int]: The set of valid token ids that can be sampled.
                    - str: The finished value if the constraint is complete, which may not be the same as the passed completion_text.
                - Any: The updated state of the constraint, which can be used to modify the behavior of future calls.
        """

    def __or__(self, other):
        from keymaker.constraints.logical import OrConstraint

        return OrConstraint([self, other])

    def __and__(self, other):
        from keymaker.constraints.logical import AndConstraint

        return AndConstraint([self, other])

    def __invert__(self):
        from keymaker.constraints.logical import NotConstraint

        return NotConstraint(self)
