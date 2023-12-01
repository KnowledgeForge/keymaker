"""Base model interface"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, ClassVar, FrozenSet, List, Optional, Set, Tuple

from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens
from keymaker.utils.general import TokenCount


class Model(ABC):
    """
    A base model from which to derive all models that generate text.
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
        token_counter: Optional[TokenCount] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the model.

        Args:
            text (str): The text to generate from.
            max_tokens (int): The maximum number of tokens in the generated text. Defaults to 1.
            selected_tokens (Optional[Set[int]]): A set of tokens that should be excluded from the generated text. Defaults to None.
            decoder (Optional[Decoder]): A parameterized description of how to select tokens from the distribution. Defaults to None.
            timeout (float): The timeout for the generation process. Defaults to 10.0.
            token_counter (Optional[TokenCount]): A counter for tracking token usage. Defaults to None.

        Yields:
            str: The generated text.
        """

    async def sample(
        self,
        text: str,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        chunk_size: Optional[int] = None,
        token_counter: Optional[TokenCount] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Sample from the language model given the input text and the selected tokens to constrain the sampling.

        Args:
            text (str): The input text to the language model.
            selected_tokens (Optional[SelectedTokens]): The set of token ids to constrain the sampling. Defaults to None.
            decoder (Optional[Decoder]): A parameterized description of how to select tokens from the distribution. Defaults to None.
            timeout (float): The timeout for the sampling process. Defaults to 10.0.
            chunk_size (Optional[int]): The number of tokens to generate in each chunk. Defaults to None.
            token_counter (Optional[TokenCount]): A counter for tracking token usage. Defaults to None.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing the generated text and the corresponding probabilities.
        """
        pre_gen_prompt_tokens = None
        pre_gen_completion_tokens = None
        if token_counter:
            pre_gen_prompt_tokens = token_counter.prompt_tokens
            pre_gen_completion_tokens = token_counter.completion_tokens

        gen = self.generate(
            text=text,
            max_tokens=(chunk_size is not None and 0 < chunk_size < self.sample_chunk_size and chunk_size)
            or self.sample_chunk_size,
            selected_tokens=selected_tokens,
            decoder=decoder,
            timeout=timeout,
            token_counter=token_counter,
        )

        ret, probs = [], []
        async for tok, prob in gen:  # type: ignore
            ret.append(tok)
            probs += prob
        if token_counter:
            if pre_gen_prompt_tokens == token_counter.prompt_tokens:
                token_counter.add_prompt_tokens(len(self.encode(text)))
            if pre_gen_completion_tokens == token_counter.completion_tokens:
                token_counter.add_completion_tokens(len(probs))
        return ret, probs

    @abstractmethod
    def encode(self, text: str) -> TokenIds:
        """
        Encode the input text as token ids.

        Args:
            text (str): The input text to encode.

        Returns:
            TokenIds: The encoded token ids.
        """

    @abstractmethod
    def decode(self, ids: TokenIds) -> str:
        """
        Decode the token ids into text.

        Args:
            ids (TokenIds): The token ids to decode.

        Returns:
            str: The decoded text.
        """

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of the language model.

        Returns:
            int: The vocabulary size.
        """
        return len(self.tokens)

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """
        Get the token id of the end of sequence (eos) token.

        Returns:
            int: The token id of the eos token.
        """

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """
        Get the token id of the beginning of sequence (bos) token.

        Returns:
            int: The token id of the bos token.
        """


class ChatModel(Model):
    """
    Base model for chat-based models.
    """
