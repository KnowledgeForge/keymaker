from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, Union

import regex as re
from keymaker.models.base import Model
from parsy import ParseError, Parser

from keymaker.constraints.base import Constraint
from keymaker.types import TokenConstraint

try:
    from lark import Lark
except ImportError:
    Lark = None

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
        for term in parser.terminals:  # type: ignore
            if term.pattern:
                regex_map[term.name] = re.compile(term.pattern.to_regexp())
        self._terminal_regexes = regex_map

    def _is_valid_token(self, patterns: List[re.Pattern[str]], token_id: int, partial_completion: str, model: "Model") -> bool:
        poss_completion = partial_completion + model.tokens[token_id]
        return any((pattern.fullmatch(poss_completion, partial=True) for pattern in patterns))

    def _compile_re(self, pattern: str) -> re.Pattern[str]:
        try:
            return re.compile(self.ignore_pattern + pattern + self.ignore_pattern)
        except Exception:
            return re.compile(self.ignore_pattern + re.escape(pattern) + self.ignore_pattern)

    async def _constrain_tokens_parsy(
        self,
        base_text: str,
        completion_text: str,
        model: "Model",
    ) -> Coroutine[Any, Any, TokenConstraint]:
        # find what rules from the parser are valid next
        try:
            self.parser.parse(completion_text)  # type: ignore
            if self.terminate_on_parse:
                return completion_text
        except ParseError as e:
            valid_next_lex, index = e.expected, e.index
            if self.terminate_on_parse:
                try:
                    test = completion_text[:index]
                    self.parser.parse(test)
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
        return valid_token_ids

    # async def _constrain_tokens_lark(
    #     self,
    #     base_text: str,
    #     completion_text: str,
    #     model: "Model",
    # ) -> Coroutine[Any, Any, TokenConstraint]:
    #     import pdb

    #     pdb.set_trace()
    #     # find what rules from the parser are valid next
    #     try:
    #         parser.parse(input_str)  # type: ignore
    #     except UnexpectedInput as e:
    #         ret = set()
    #         try:
    #             ret |= set(e.expected)
    #         except AttributeError:
    #             pass
    #         try:
    #             ret |= set(e.allowed)
    #         except AttributeError:
    #             pass
    #         return ret
    #     return set()

    #     if len(valid_next_lex) == 0:
    #         return set(), None
    #     if len(valid_next_lex) == 1 and '$END' in valid_next_lex:
    #         return completion_text, None

    #     last_lex = state['last_lex'] if state else set()
    #     start_idx = state['start_idx'] if state else 0

    #     # strip that last lex if there is a match
    #     # our parser regex patterns only tell us the next completion pattern
    #     # and cannot be complete of what we have already selected
    #     for lex in last_lex:
    #         if self.terminal_regexes[lex].fullmatch(completion_text[start_idx:]):
    #             start_idx = len(completion_text)
    #             last_lex = set()
    #             break

    #     regex_pattern = [self.terminal_regexes[t] for t in valid_next_lex | last_lex]

    #     with ThreadPoolExecutor():
    #         valid_token_ids = set(
    #             filter(
    #                 lambda token_id: self._is_valid_token(regex_pattern, token_id, completion_text[start_idx:], model),
    #                 model.tokens.keys(),
    #             ),
    #         )

    #     return valid_token_ids

    async def constrain_tokens(self, base_text: str, completion_text: str, model: Model) -> Coroutine[Any, Any, Coroutine[Any, Any, TokenConstraint]]:
        if isinstance(self.parser, Parser):
            return await self._constrain_tokens_parsy(base_text, completion_text, model)
        return await self._constrain_tokens_lark(base_text, completion_text, model)


### LARK BASED
# from lark import Lark, UnexpectedInput
# @dataclass
# class ParserConstraint(Constraint):
#     """Constrain token ids that can be sampled based on a context-free grammar or a pre-built parser.

#     Attributes:
#         grammar (str, optional): The context-free grammar to use for parsing. Either the grammar or a pre-built parser
#             must be provided.
#         parser (Lark, optional): A pre-built parser that will be used for parsing. Either the grammar or a pre-built
#             parser must be provided.

#     Notes:
#         The `parser` attribute takes precedence over `grammar` if both are provided.
#         Based on https://github.com/r2d4/parserllm.
#     """

#     grammar: Optional[str] = None
#     parser: Optional[Lark] = None
#     terminal_terminate_on_match: Dict[str, bool] = field(default_factory=lambda: dict(CNAME=False))

#     def __post_init__(self):
#         if self.grammar and self.parser:
#             raise ValueError("Provided values for both `grammar` and `parser`. Choose only one.")
#         if not (self.grammar or self.parser):
#             raise ValueError("Must provide one of `grammar` or `parser`.")
#         if self.grammar:
#             self.parser = Lark(self.grammar, parser = 'lalr')

#         self.terminal_regexes = self.extract_terminal_regex(self.parser)

#     @staticmethod
#     def extract_terminal_regex(parser: Lark) -> Dict[str, re.Pattern]:
#         regex_map = {}
#         for term in parser.terminals:  # type: ignore
#             if term.pattern:
#                 regex_map[term.name] = re.compile(term.pattern.to_regexp())
#         return regex_map

#     @staticmethod
#     def next_lex(input_str: str, parser: Lark) -> Set[str]:
#         try:
#             parser.parse(input_str)  # type: ignore
#         except UnexpectedInput as e:
#             ret = set()
#             try:
#                 ret |= set(e.expected)
#             except AttributeError:
#                 pass
#             try:
#                 ret |= set(e.allowed)
#             except AttributeError:
#                 pass
#             return ret
#         return set()

#     def _is_valid_token(self, patterns: List[re.Pattern[str]], token_id: int, partial_completion: str, model: "Model") -> bool:
#         poss_completion = partial_completion + model.tokens[token_id]
#         return any((pattern.fullmatch(poss_completion, partial=True) for pattern in patterns))

#     def constrain_tokens(
#         self,
#         base_text: str,
#         completion_text: str,
#         model: "Model",
#         state: Optional[Any] = None,
#     ) -> Tuple[TokenConstraint, Any]:
#         import pdb; pdb.set_trace()
#         # find what rules from the parser are valid next
#         valid_next_lex = self.next_lex(completion_text, self.parser)
#         if len(valid_next_lex) == 0:
#             return set(), None
#         if len(valid_next_lex) == 1 and '$END' in valid_next_lex:
#             return completion_text, None

#         last_lex = state['last_lex'] if state else set()
#         start_idx = state['start_idx'] if state else 0

#         # strip that last lex if there is a match
#         # our parser regex patterns only tell us the next completion pattern
#         # and cannot be complete of what we have already selected
#         for lex in last_lex:
#             if self.terminal_regexes[lex].fullmatch(completion_text[start_idx:]):
#                 start_idx = len(completion_text)
#                 last_lex = set()
#                 break

#         regex_pattern = [self.terminal_regexes[t] for t in valid_next_lex | last_lex]

#         with ThreadPoolExecutor():
#             valid_token_ids = set(
#                 filter(
#                     lambda token_id: self._is_valid_token(regex_pattern, token_id, completion_text[start_idx:], model),
#                     model.tokens.keys(),
#                 ),
#             )

#         return valid_token_ids, {'start_idx': start_idx, 'last_lex': valid_next_lex}
