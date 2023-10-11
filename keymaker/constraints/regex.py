"""Constraints for regex patterns"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Union

import regex as re

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint


@dataclass
class RegexConstraint(Constraint):
    """Constrain token ids that can be sampled based on a regex pattern.

    Attributes:
        pattern (str): The regex pattern to match.

    Notes:
        Based on https://github.com/r2d4/rellm
    """

    pattern: Union[str, re.Pattern]
    terminate_on_match: bool = True

    def __post_init__(self):
        self._pattern: re.Pattern = re.compile(self.pattern) if isinstance(self.pattern, str) else self.pattern

    def _is_valid_token(self, token_id: int, partial_completion: str, model: "Model") -> bool:
        decoded_token = model.tokens[token_id]
        return self._pattern.fullmatch(partial_completion + decoded_token, partial=True)

    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: "Model",
    ) -> TokenConstraint:
        add_eos = False
        match = self._pattern.match(completion_text)
        if match and match.start() == 0:
            if self.terminate_on_match:
                return completion_text[: match.end()]
            else:
                add_eos = True

        with ThreadPoolExecutor():
            valid_token_ids = set(
                filter(
                    lambda token_id: self._is_valid_token(token_id, completion_text, model),
                    model.tokens.keys(),
                ),
            )
        if add_eos:
            valid_token_ids.add(model.eos_token_id)
        return valid_token_ids
