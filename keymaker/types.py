'''Common types used throughout keymaker'''
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Union


class DecodingStrategy(str, Enum):
    GREEDY = "GREEDY"
    SAMPLE = "SAMPLE"


@dataclass
class Decoder:
    temperature: float = 0.7
    top_p: float = 0.95
    strategy: DecodingStrategy = DecodingStrategy.GREEDY


TokenIds = List[int]
Tokens = Dict[int, str]
TokenDistribution = Dict[str, int]
SelectedTokens = Set[int]
TokenConstraint = Union[None, SelectedTokens, str]
