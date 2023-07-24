"""Constraints over sets of fixed options"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Set

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint


@dataclass
class OptionsConstraint(Constraint):
    """
    Options constraint constrains output based on a list of string options
    """

    options: Set[str]
    short_circuit: bool = True  # early return when available options based on completed text are <=1

    def _is_valid_token(self, token_id: int, partial_completion: str, model: Model) -> bool:
        decoded_token = model.tokens[token_id]
        return any(option.startswith(partial_completion + decoded_token) for option in self.options)

    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: Model,
    ) -> TokenConstraint:
        if completion_text in self.options:
            return completion_text

        if completion_text and self.short_circuit:
            limited_options = set()
            for option in self.options:
                if option.startswith(completion_text):
                    limited_options.add(option)
                    if len(limited_options) > 1:
                        break
            if len(limited_options) == 0:
                return set()
            if len(limited_options) == 1:
                return limited_options.pop()

        with ThreadPoolExecutor():
            valid_token_ids = set(
                filter(
                    lambda token_id: self._is_valid_token(token_id, completion_text, model),
                    model.tokens.keys(),
                ),
            )

        return valid_token_ids
