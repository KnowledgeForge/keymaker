"""Base model interface"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, ClassVar, FrozenSet, List, Optional, Set, Tuple

from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens


class Model(ABC):
    """
    A base model from which to derive all models that generate text
    """

    tokens: Tokens
    max_total_tokens: int = 512
    supported_decodings: ClassVar[FrozenSet[DecodingStrategy]]
    sample_chunk_size: int = 1

    @abstractmethod
    async def generate(
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[Set[int]] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the Huggingface model.

        Args:
            text: The text to generate from.
            max_length: The maximum length of the generated text.
            selected_tokens: A set of tokens that should be excluded from the generated text.
            decoder: A parameterized description of how to select tokens from the distribution
            timeout: The timeout for the generation process.

        Returns:
            An iterator of generated text.
        """

    async def sample(
        self,
        text: str,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> Tuple[List[str], List[float]]:
        """Sample from the language model given the input text and the selected tokens to constrain the sampling.

        Args:
            text (str): The input text to the language model.
            selected_tokens (Optional[Set[int]]): The set of token ids to constrain the sampling. Defaults to None.

        Returns:
            str: The generated text from the language model.
        """
        gen = self.generate(
            text=text,
            max_tokens=self.sample_chunk_size,
            selected_tokens=selected_tokens,
            decoder=decoder,
            timeout=timeout,
        )
        ret, probs = [], []
        async for tok, prob in gen:  # type: ignore
            ret.append(tok)
            probs += prob
        return ret, probs

    @abstractmethod
    def encode(self, text: str) -> TokenIds:
        """Encode the input text as token ids.

        Args:
            text (str): The input text to encode.

        Returns:
            TokenIds: The encoded token ids.
        """

    @abstractmethod
    def decode(self, ids: TokenIds) -> str:
        """Decode the token ids into text.

        Args:
            ids (TokenIds): The token ids to decode.

        Returns:
            str: The decoded text.
        """

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the language model."""
        return len(self.tokens)

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Get the token id of the end of sequence (eos) token."""

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Get the token id of the beginning of sequence (bos) token."""


class ChatModel(Model):
    """Base model for chat based models"""
