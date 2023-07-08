from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import regex as re
from lark import Lark, UnexpectedInput

from keymaker.constraints.base import Constraint
from keymaker.constraints.logical import OrConstraint
from keymaker.constraints.regex import RegexConstraint
from keymaker.types import TokenConstraint

if TYPE_CHECKING:
    from keymaker.models.base import Model


@dataclass
class ParserConstraint(Constraint):
    """Constrain token ids that can be sampled based on a context-free grammar or a pre-built parser.

    Attributes:
        grammar (str, optional): The context-free grammar to use for parsing. Either the grammar or a pre-built parser
            must be provided.
        parser (Lark, optional): A pre-built parser that will be used for parsing. Either the grammar or a pre-built
            parser must be provided.

    Notes:
        The `parser` attribute takes precedence over `grammar` if both are provided.
        Based on https://github.com/r2d4/parserllm.
    """

    grammar: Optional[str] = None
    parser: Optional[Lark] = None

    def __post_init__(self):
        if self.grammar and self.parser:
            raise ValueError("Provided values for both `grammar` and `parser`. Choose only one.")
        if not (self.grammar or self.parser):
            raise ValueError("Must provide one of `grammar` or `parser`.")
        if self.grammar:
            self.parser = Lark(self.grammar)

        self.terminal_regexes = self.extract_terminal_regex(self.parser)

    @staticmethod
    def extract_terminal_regex(parser: Lark) -> Dict[str, re.Pattern]:
        regex_map = {}
        for term in parser.terminals:  # type: ignore
            if term.pattern:
                regex_map[term.name] = re.compile(term.pattern.to_regexp())
        return regex_map

    @staticmethod
    def next_lex(input_str: str, parser: Lark) -> List[str]:
        try:
            parser.parse(input_str)  # type: ignore
        except UnexpectedInput as e:
            expected_tokens = e.expected
            parser.last_expected = expected_tokens
            return expected_tokens
        return []

    def constrain_tokens(self, base_text: str, completion_text: str, model: "Model") -> TokenConstraint:

        valid_next_lex = self.next_lex(completion_text, self.parser)
        if len(valid_next_lex) == 0:
            return set()
        if ['$END'] == valid_next_lex:
            return completion_text

        regex_constraints = [RegexConstraint(self.terminal_regexes[t]) for t in valid_next_lex] + [
            RegexConstraint(model.decode([model.eos_token_id])),
        ]

        constraint = OrConstraint(regex_constraints)

        valid_token_ids = constraint.constrain_tokens(base_text, completion_text, model)

        return valid_token_ids
