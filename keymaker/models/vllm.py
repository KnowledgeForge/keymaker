"""An implementation of a model for served vLLM"""

from dataclasses import dataclass
from typing import AsyncGenerator, FrozenSet, List, Optional
from keymaker.models.huggingface import transformers_tokens
from keymaker.models.openai import OpenAICompletion
from transformers import AutoTokenizer
from keymaker.types import  DecodingStrategy, TokenIds
import openai


@dataclass
class VLLM(OpenAICompletion):
    """
    A VLLM model behind an api. Uses a huggingface tokenizer
    """

    model_name: str
    api_key: str = "EMPTY"
    api_base: str = "http://localhost:8000/v1"
    tokenizer: Optional[AutoTokenizer] = None
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )

    def __post_init__(self):
        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(self.model_name)
        self.tokens = transformers_tokens(self.tokenizer)
        self._api_base = openai.api_base
        self._api_key = openai.api_key

    def encode(self, text: str) -> TokenIds:
        return self.tokenizer.encode(text)  # type: ignore

    def decode(self, ids: TokenIds) -> str:
        return self.tokenizer.decode(ids)  # type: ignore

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id  # type: ignore

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id  # type: ignore

    async def generate(  # type: ignore
        self,
        *args, **kwargs
    ) -> AsyncGenerator[str, None]:
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        ret = super().generate(*args, **kwargs)
        # doubtful this will work since the generate call is async
        openai.api_key = self._api_key
        openai.api_base = self._api_base
        return ret