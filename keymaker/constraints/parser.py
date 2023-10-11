from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Union

import regex as re
from parsy import ParseError, Parser

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint

try:
    from lark import Lark, UnexpectedCharacters, UnexpectedEOF
except ImportError:
    Lark = None


@dataclass
class ParserConstraint(Constraint):
    """Constrain token ids that can be sampled based on a context-free grammar or a pre-built parser.

    Attributes:
        grammar (str, optional): The context-free grammar to use for parsing. Either the grammar or a pre-built parser
            must be provided.
        parser (Lark, optional): A pre-built parser that will be used for parsing. Either the grammar or a pre-built
            parser must be provided.

    """

    parser: Optional[Union[Parser, "Lark"]] = None
    terminate_on_parse: bool = True
    ignore_pattern: str = r"\s*"

    def __post_init__(self):
        self._terminal_regexes = None
        if Lark is not None and isinstance(self.parser, Lark):
            self._extract_terminal_regex()

    def _extract_terminal_regex(self):
        regex_map = {}
        for term in self.parser.terminals:  # type: ignore
            if term.pattern:
                regex_map[term.name] = term.pattern.to_regexp()
        self._terminal_regexes = regex_map

    def _is_valid_token(self, patterns: List[re.Pattern], token_id: int, partial_completion: str, model: "Model") -> bool:
        poss_completion = partial_completion + model.tokens[token_id]
        return any((pattern.fullmatch(poss_completion, partial=True) for pattern in patterns))

    def __hash__(self) -> int:
        return id(self)

    @lru_cache
    def _compile_re(self, pattern: str) -> re.Pattern:
        try:
            return re.compile(self.ignore_pattern + pattern + self.ignore_pattern)
        except Exception:
            return re.compile(self.ignore_pattern + re.escape(pattern) + self.ignore_pattern)

    async def _constrain_tokens_parsy(
        self,
        base_text: str,
        completion_text: str,
        model: "Model",
    ) -> TokenConstraint:
        # find what rules from the parser are valid next
        add_eos = False
        try:
            self.parser.parse(completion_text)  # type: ignore
            if self.terminate_on_parse:
                return completion_text
            else:
                add_eos = True
        except ParseError as e:
            valid_next_lex, index = e.expected, e.index
            if self.terminate_on_parse:
                try:
                    test = completion_text[:index]
                    self.parser.parse(test)  # type: ignore
                    return test
                except Exception:
                    pass

        regex_patterns = []
        for lex in valid_next_lex:
            regex_patterns.append(self._compile_re(lex))

        with ThreadPoolExecutor():
            valid_token_ids = set(
                filter(
                    lambda token_id: self._is_valid_token(regex_patterns, token_id, completion_text[index:], model),
                    model.tokens.keys(),
                ),
            )
        if add_eos:
            valid_token_ids.add(model.eos_token_id)
        return valid_token_ids

    async def _constrain_tokens_lark(
        self,
        base_text: str,
        completion_text: str,
        model: "Model",
    ) -> TokenConstraint:
        index = None
        add_eos = False
        try:
            self.parser.parse(completion_text)  # type: ignore
            if self.terminate_on_parse:
                return completion_text
            else:
                add_eos = True
        except UnexpectedCharacters as e:
            valid_next_lex = {self._terminal_regexes[lex] for lex in e.allowed}
            index = e.pos_in_stream

        except UnexpectedEOF as e:
            valid_next_lex = {self._terminal_regexes[lex] for lex in e.expected}
            index = len(completion_text)

        if index and self.terminate_on_parse:
            try:
                test = completion_text[:index]
                self.parser.parse(test)  # type: ignore
                return test
            except Exception:
                pass

        regex_patterns = [self._compile_re(lex) for lex in valid_next_lex]

        with ThreadPoolExecutor():
            valid_token_ids = set(
                filter(
                    lambda token_id: self._is_valid_token(regex_patterns, token_id, completion_text[index:], model),
                    model.tokens.keys(),
                ),
            )
        if add_eos:
            valid_token_ids.add(model.eos_token_id)
        return valid_token_ids

    async def constrain_tokens(
        self,
        base_text: str,
        completion_text: str,
        model: Model,
    ) -> TokenConstraint:
        if isinstance(self.parser, Parser):
            return await self._constrain_tokens_parsy(base_text, completion_text, model)
        return await self._constrain_tokens_lark(base_text, completion_text, model)
