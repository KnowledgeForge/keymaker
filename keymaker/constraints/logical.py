"""Common logical constraints for combining other constraints"""



@dataclass
class NotConstraint(Constraint):
    """Invert a token id constraint.

    Attributes:
        constraint (Constraint): The constraint to negate.
    """

    constraint: Constraint

    def constrain_tokens(
        self, base_text: str, completion_text: str, model: "Model"
    ) -> TokenConstraint:
        selected_tokens = self.constraint.constrain_tokens(
            base_text, completion_text, model
        )
        if selected_tokens is None or isinstance(selected_tokens, str):
            return selected_tokens
        return {tok for tok in model.tokens if tok not in selected_tokens}


@dataclass
class AndConstraint(Constraint):
    """Constrain token ids that can be sampled by applying multiple constraints.

    Attributes:
        constraints (List[Constraint]): The list of constraints to apply.
    """

    constraints: List[Constraint]

    def constrain_tokens(
        self, base_text: str, completion_text: str, model: "Model"
    ) -> Union[None, Set[int], str]:
        ret = None
        completions = []
        for constraint in self.constraints:
            completions = []
            selected_tokens = constraint.constrain_tokens(
                base_text, completion_text, model
            )
            if selected_tokens is None:
                # Do nothing because all tokens are valid
                pass
            if isinstance(selected_tokens, str):
                completions.append(selected_tokens)
            if isinstance(selected_tokens, set):
                ret = ret & selected_tokens if ret is not None else selected_tokens
        if len(completions) == len(self.constraints):
            if len(set(completions)) != 1:
                raise ValueError(
                    f"Got different completions for constraints `{self}`. Completions: `{set(completions)}`"
                )
            return completions[0]
        return ret


@dataclass
class OrConstraint(Constraint):
    """Constrain token ids that can be sampled by applying multiple constraints.

    Attributes:
        constraints (List[Constraint]): The list of constraints to apply.
    """

    constraints: List[Constraint]

    def constrain_tokens(
        self, base_text: str, completion_text: str, model: "Model"
    ) -> Union[None, Set[int], str]:
        ret = set()
        for constraint in self.constraints:
            selected_tokens = constraint.constrain_tokens(
                base_text, completion_text, model
            )
            if selected_tokens is None:
                # One allows everything so overall the or does
                return None
            if isinstance(selected_tokens, str):
                return selected_tokens
            if isinstance(selected_tokens, set):
                ret |= selected_tokens
        return ret