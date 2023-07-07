"""Models to use with any openai compatible api endpoints"""
import warnings
from dataclasses import dataclass
from typing import Any, AsyncGenerator, FrozenSet, Optional

import aiohttp
import openai
import tiktoken
from tiktoken import Encoding

from keymaker.models.base import Model
from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens
from keymaker.utils.chat import split_tags


def openai_tokens(tokenizer: Encoding) -> Tokens:
    vocab_len = len(tokenizer.token_byte_values())
    tokens = {i: tokenizer.decode([i]) for i in range(vocab_len - 1)}
    for i in range(vocab_len, tokenizer.max_token_value):
        tokens[i] = f"<|special_{i}|>"
    return tokens


@dataclass
class OpenAIChat(Model):
    model_name: str = "gpt-3.5-turbo"
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )
    role_tag_start: str = "%"
    role_tag_end: str = "%"
    default_role: str = "assistant"
    allowed_roles: FrozenSet[str] = frozenset(("system", "user", "assistant"))
    max_retries: int = 10
    retry_sleep_time: float = 1.0
    max_token_selection: int = 300

    def __post_init__(self):
        self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.tokens = openai_tokens(self._tokenizer)

    def encode(self, text: str) -> TokenIds:
        return self._tokenizer.encode(text)

    def decode(self, ids: TokenIds) -> str:
        return self._tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eot_token

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.encode_single_token("<|endofprompt|>")

    async def _generate(
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[Any, None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(f"Unsupported decoding strategy for {self.__class__} model `{decoder.strategy}`.")
        messages = split_tags(
            text,
            self.role_tag_start,
            self.role_tag_end,
            self.default_role,
            self.allowed_roles,
        )

        temperature = decoder.temperature
        top_p = decoder.top_p
        if decoder.strategy == DecodingStrategy.GREEDY:
            # try to make the sampling as deterministic as possible
            # to select only the one top token
            top_p = 0.01  # select only n tokens to get over .01, should virually always be a single token
            temperature = 0.0

        selected_tokens = selected_tokens or set()
        payload = {
            "messages": messages,
            "logit_bias": {str(token): 100 for token in selected_tokens},
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with aiohttp.ClientSession() as session:
            openai.aiosession.set(session)
            completion_stream = await openai.ChatCompletion.acreate(**payload, stream=True)
            async for chat_completion in completion_stream:
                yield chat_completion

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        if selected_tokens and len(selected_tokens) > self.max_token_selection:
            warnings.warn(
                f"Trying to mask {len(selected_tokens)} tokens which "
                f"is more than {self.max_token_selection} mask limit "
                f"of {self}. Consider stricter constraints. Will select"
                "lowest token ids up to this limit.",
            )
            selected_tokens = list(selected_tokens)[: self.max_token_selection]  # type: ignore

        def result_handler(response):
            delta = response["choices"][0]["delta"]
            return (
                "" if "content" not in delta else delta["content"],  # content
                "finish_reason" in delta and delta["finish_reason"] is not None,  # complete generation
            )

        eos_token = self.decode([self.eos_token_id])
        error = False
        retries = 0
        for retries in range(self.max_retries):
            async for chat_completion in self._generate(
                text=text,
                max_tokens=max_tokens,
                selected_tokens=selected_tokens,
                decoder=decoder,
                timeout=timeout,
            ):
                if "error" in chat_completion.keys():
                    message = chat_completion["error"]["message"]
                    retry = retries < self.max_retries
                    retries += 1
                    warnings.warn(
                        "OpenAI Chat Completion API raised an error: \n" f"MESSAGE: {message}\n" f"RETRYING {retries}"
                        if retry
                        else "",
                    )
                    error = True
                    break
                else:
                    error = False
                    content, done = result_handler(chat_completion)
                    text += content
                    if content.startswith(eos_token):
                        break
                    if content:
                        yield content
                    if done:
                        break
            if not error:
                break


@dataclass
class OpenAICompletion(Model):
    model_name: str = "text-ada-001"
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )
    role_tag_start: str = "%"
    role_tag_end: str = "%"
    default_role: str = "assistant"
    allowed_roles: FrozenSet[str] = frozenset(("system", "user", "assistant"))
    max_retries: int = 10
    retry_sleep_time: float = 1.0
    max_token_selection: int = 300

    def __post_init__(self):
        self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.tokens = openai_tokens(self._tokenizer)

    def encode(self, text: str) -> TokenIds:
        return self._tokenizer.encode(text)

    def decode(self, ids: TokenIds) -> str:
        return self._tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eot_token

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.encode_single_token("<|endofprompt|>")

    async def _generate(
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(f"Unsupported decoding strategy for {self.__class__} model `{decoder.strategy}`.")

        temperature = decoder.temperature
        top_p = decoder.top_p
        if decoder.strategy == DecodingStrategy.GREEDY:
            # try to make the sampling as deterministic as possible
            # to select only the one top token
            top_p = 0.01  # select only n tokens to get over .01, should virually always be a single token
            temperature = 0.0

        selected_tokens = selected_tokens or set()
        payload = {
            "prompt": text,
            "logit_bias": {str(token): 100 for token in selected_tokens},
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with aiohttp.ClientSession() as session:
            openai.aiosession.set(session)
            completion_stream = await openai.Completion.acreate(**payload, stream=True)

            async for completion in completion_stream:
                yield completion

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[Any, None]:
        if selected_tokens and len(selected_tokens) > self.max_token_selection:
            warnings.warn(
                f"Trying to mask {len(selected_tokens)} tokens which "
                f"is more than {self.max_token_selection} mask limit "
                f"of {self}. Consider stricter constraints. Will select"
                "lowest token ids up to this limit.",
            )
            selected_tokens = list(selected_tokens)[: self.max_token_selection]  # type: ignore

        def result_handler(response):
            delta = response.choices[0]
            return (
                "" if "text" not in delta else delta["text"],  # content
                "finish_reason" in delta and delta["finish_reason"] is not None,  # complete generation
            )

        eos_token = self.decode([self.eos_token_id])
        error = False
        retries = 0
        for retries in range(self.max_retries):
            async for completion in self._generate(
                text=text,
                max_tokens=max_tokens,
                selected_tokens=selected_tokens,
                decoder=decoder,
                timeout=timeout,
            ):
                if "error" in completion.keys():  # type: ignore
                    message = completion["error"]["message"]  # type: ignore
                    retry = retries < self.max_retries
                    retries += 1
                    warnings.warn(
                        "OpenAI Completion API raised an error: \n" f"MESSAGE: {message}\n" f"RETRYING {retries}"
                        if retry
                        else "",
                    )
                    error = True
                    break
                else:
                    error = False
                    content, done = result_handler(completion)
                    text += content
                    if content.startswith(eos_token):
                        break
                    if content:
                        yield content
                    if done:
                        break
            if not error:
                break
