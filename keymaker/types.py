"""Common types used throughout keymaker"""
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Protocol, Set, Union

if TYPE_CHECKING:
    from keymaker.prompt import CompletionConfig, Prompt


class DecodingStrategy(str, Enum):
    GREEDY = "GREEDY"
    SAMPLE = "SAMPLE"


@dataclass
class Decoder:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    strategy: DecodingStrategy = DecodingStrategy.GREEDY

    def __post_init__(self):
        if self.top_k is not None and self.top_k != 1 and self.strategy == DecodingStrategy.GREEDY:
            warnings.warn(f"Greedy decoding top_k {self.top_k} ignored. Setting top_k to 1.")
            self.top_k = 1


TokenIds = List[int]
Tokens = Dict[int, str]
TokenDistribution = Dict[str, int]
SelectedTokens = Set[int]
TokenConstraint = Union[None, SelectedTokens, str]


class Stringable(Protocol):
    def __str__(self) -> str:
        pass


FormatArg = Union[
    Stringable,
    Callable[["Prompt"], Union[Stringable, "CompletionConfig"]],
    Callable[["Prompt"], Iterable[Union[Stringable, "CompletionConfig"]]],
]
