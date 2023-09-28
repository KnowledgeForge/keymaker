"""A parse constraint which gives correct JSON"""

from dataclasses import dataclass
from typing import Optional, Set

from parsy import alt, forward_declaration, regex, seq, string

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import TokenConstraint

# parser derived from parsy example
# Utilities
whitespace = regex(r"\s+")
optional_whitespace = lambda p: p | (whitespace >> p) | (p << whitespace)  # noqa: E731

# Punctuation
lbrace = optional_whitespace(string("{"))
rbrace = optional_whitespace(string("}"))
lbrack = optional_whitespace(string("["))
rbrack = optional_whitespace(string("]"))
colon = optional_whitespace(string(":"))
comma = optional_whitespace(string(","))

# Primitives
true = optional_whitespace(string("true")).result(True)
false = optional_whitespace(string("false")).result(False)
null = optional_whitespace(string("null")).result(None)
number = optional_whitespace(regex(r"-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?")).map(float)
string_part = regex(r'[^"\\]+')
string_esc = string("\\") >> (
    string("\\")
    | string("/")
    | string('"')
    | string("b").result("\b")
    | string("f").result("\f")
    | string("n").result("\n")
    | string("r").result("\r")
    | string("t").result("\t")
    | regex(r"u[0-9a-fA-F]{4}").map(lambda s: chr(int(s[1:], 16)))
)
quoted = optional_whitespace(string('"') >> (string_part | string_esc).many().concat() << string('"'))


@dataclass
class JsonConstraint(Constraint):
    keys: Optional[Set[str]] = None
    allow_array: bool = True
    allow_null: bool = True
    allow_string: bool = True
    allow_number: bool = True
    allow_boolean: bool = True

    def __post_init__(self):
        if self.keys:
            key = optional_whitespace(string('"') >> (alt(*[string(key for key in self.keys)])) << string('"'))
        else:
            key = optional_whitespace(string('"') >> (string_part | string_esc).at_least(1).concat() << string('"'))

        # Data structures
        json_value = forward_declaration()
        object_pair = seq(key << colon, json_value).map(tuple)
        json_object = lbrace >> object_pair.sep_by(comma).map(dict) << rbrace
        array = lbrack >> json_value.sep_by(comma) << rbrack

        opts = []
        if self.allow_string:
            opts.append(quoted)
        if self.allow_number:
            opts.append(number)
        opts.append(json_object)
        if self.allow_array:
            opts.append(array)
        if self.allow_boolean:
            opts.append(true)
            opts.append(false)
        if self.allow_null:
            opts.append(null)

        # Everything
        json_value.become(alt(*opts))
        json_doc = json_value

        from keymaker.constraints import ParserConstraint

        self.constraint = ParserConstraint(parser=json_doc)

    async def constrain_tokens(self, base_text: str, completion_text: str, model: Model) -> TokenConstraint:
        return await self.constraint.constrain_tokens(base_text, completion_text, model)
