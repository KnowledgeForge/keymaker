"""A constraint based on some model's output"""

from dataclasses import dataclass
from typing import Optional, Set

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint


@dataclass
class ModelConstraint(Constraint):
    """Use another model to constrain token output. This model must implement `probs`.
    The model should be able to return the full token distribution.

    Attributes:
        model (Model): a model that supports the `probs` method
        top_k: Optional[int]: number of tokens to select from this model to filter by
        top_p: Optional[float]: probability tokens can sum to to select
        constraint: Optional[Constraint]: another constraint to apply to restrict this models output

    """

    model: Model
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    constraint: Optional[Constraint] = None

    def __post_init__(self):
        if self.top_k is None and self.top_p is None:
            raise ValueError("At least one of top_k and top_p must be set.")

    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: Model,
    ) -> TokenConstraint:
        preselected: Optional[Set[int]] = None

        if self.constraint is not None:
            preselected = await self.constraint.constrain_tokens(base_text, completion_text, model)  # type: ignore
            if not isinstance(preselected, set) or len(preselected) == 0:
                return set()

        if self.top_k and preselected and len(preselected) <= self.top_k:
            return preselected

        if not hasattr(self.model, "probs"):
            raise TypeError(f"Model constraint requires the model to implement `probs`. Got {self.model}.")
        probs = sorted((await self.model.probs(base_text + completion_text)), key=lambda e: e[1], reverse=True)  # type: ignore
        selected_tokens = set()
        tot_p = 0
        tot_k = 0
        for tok, p in probs:
            if self.top_p and tot_p > self.top_p:
                break
            if self.top_k and tot_k >= self.top_k:
                break
            try:
                model_tok_id = model.encode(tok)[0]
            except Exception:
                continue
            if preselected and model_tok_id not in preselected:
                continue
            if model_tok_id not in selected_tokens:
                selected_tokens.add(model_tok_id)
                tot_p += p
                tot_k += 1
        return selected_tokens
