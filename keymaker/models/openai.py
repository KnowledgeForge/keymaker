"""Models to use with any openai compatible api endpoints"""
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, FrozenSet, List, Optional, Tuple

import openai
import tiktoken
from tiktoken import Encoding

from keymaker.models.base import ChatModel, Model
from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens
from keymaker.utils.chat import split_tags
from keymaker.utils.exceptions import AggregateException, Deprecation
from keymaker.utils.general import TokenCount


@lru_cache(10)
def openai_tokens(tokenizer: Encoding) -> Tokens:
    vocab_len = len(tokenizer.token_byte_values())
    tokens = {i: tokenizer.decode([i]) for i in range(vocab_len - 1)}
    for i in range(vocab_len, tokenizer.max_token_value):
        tokens[i] = f"<|special_{i}|>"
    return tokens


def model_encoding(model: str):
    if model == 'chatgpt' or model.startswith('gpt-3'):
        return 'gpt-3.5-turbo'
    if model.startswith('gpt-4'):
        return 'gpt-4'
    return model


@dataclass
class OpenAIChat(ChatModel):
    """Class representing all OpenAI API conformant chat model usage."""

    client: openai.AsyncClient = field(default_factory=openai.AsyncOpenAI)
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
    max_total_tokens: int = 4000
    sample_chunk_size: int = 12  # tokens generated by `generate` on behalf of `sample`
    addtl_create_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._tokenizer = tiktoken.encoding_for_model(model_encoding(self.model_name))
        self.tokens = openai_tokens(self._tokenizer)
        self._all_token_ids = set(self.tokens.keys())

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
        logit_bias: Optional[Dict[str, int]] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        token_counter: Optional[TokenCount] = None,
        gen_kwargs: Optional[dict] = None,
        stream: bool = True,
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

        if messages[-1]["role"] != self.default_role:
            messages.append({"role": self.default_role, "content": " "})

        if token_counter is not None:
            tot_text = "".join(m["content"] for m in messages)
            token_counter.add_prompt_tokens(len(self.encode(tot_text)))

        gen_kwargs = gen_kwargs or {}
        if temperature := decoder.temperature:
            gen_kwargs["temperature"] = temperature
        if top_p := decoder.top_p:
            gen_kwargs["top_p"] = top_p
        if top_k := decoder.top_k:
            gen_kwargs["top_k"] = top_k
        if decoder.strategy == DecodingStrategy.GREEDY:
            # try to make the sampling as deterministic as possible
            # to select only the one top token
            gen_kwargs["temperature"] = 0
            gen_kwargs["top_p"] = (
                1 / self.vocab_size
            )  # select only n tokens to get over 1/self.vocab_size, should always be a single token
            if (
                "top_k" in gen_kwargs
            ):  # non-openai yet openai-compliant apis may also support topk while the official api does not
                gen_kwargs["top_k"] = 1

        payload = {
            "messages": messages,
            "logit_bias": logit_bias,
            "model": self.model_name,
            "max_tokens": max_tokens,
            **gen_kwargs,  # type: ignore
        }

        completion_stream = await self.client.chat.completions.create(**self.addtl_create_kwargs, **payload, stream=True)
        async for chat_completion in completion_stream:
            yield chat_completion

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        token_counter: Optional[TokenCount] = None,
        gen_kwargs: Optional[dict] = None,
        stream: bool = True,
    ) -> AsyncGenerator[Tuple[str, List[None]], None]:
        bias = 100
        # if there are more tokens to keep than ignore, invert the bias
        if selected_tokens and len(selected_tokens) > (self.vocab_size / 2):
            selected_tokens = self._all_token_ids - selected_tokens
            bias = -100

        if selected_tokens and len(selected_tokens) > self.max_token_selection:
            warnings.warn(
                f"Trying to mask {len(selected_tokens)} tokens which "
                f"is more than {self.max_token_selection} mask limit "
                f"of {self}. Consider stricter constraints. Will select"
                "lowest token ids up to this limit.",
            )
            selected_tokens = sorted(list(selected_tokens))[: self.max_token_selection]  # type: ignore
        logit_bias = {}
        if selected_tokens:
            for i, idx in enumerate(selected_tokens):
                logit_bias[str(idx)] = bias

        def result_handler(response):
            choice = response.choices and response.choices[0]
            content = choice and choice.delta.content or ''
            finish_reason = choice and choice.finish_reason or None
            return (
                content,  # content
                finish_reason is not None,  # complete generation
            )

        errors = []
        for retries in range(self.max_retries):
            try:
                async for chat_completion in self._generate(
                    text=text,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                    decoder=decoder,
                    timeout=timeout,
                    token_counter=token_counter,
                    gen_kwargs=gen_kwargs,
                ):
                    content, done = result_handler(chat_completion)
                    if done:
                        break
                    if content:
                        yield (content, [None])

            except openai.APIConnectionError as exc:
                errors.append(exc)
                warnings.warn(
                    "OpenAI Chat Completion API raised an error: \n" f"MESSAGE: {exc}\n" f"RETRYING {retries}"
                    if (retries + 1) < self.max_retries
                    else "",
                )

            if not errors:
                break
            else:
                raise AggregateException(errors)


@dataclass
class OpenAICompletion(Model):
    """Class representing all OpenAI API conformant completion model usage."""

    client: openai.AsyncClient = field(default_factory=openai.AsyncOpenAI)
    model_name: str = "gpt-3.5-turbo-instruct"
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
    max_total_tokens: int = 4096
    addtl_create_kwargs: Dict[str, Any] = field(default_factory=dict)
    ignore_deprecation: bool = False

    def __post_init__(self):
        if not self.ignore_deprecation:
            raise Deprecation(
                "OpenAI has announced the end of their completions API 1/4/24. "
                "This Keymaker Model class is currently deprecated and use at your own risk until further notice."
                "Set `ignore_deprecation = True` to bypass this.",
            )
        self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.tokens = openai_tokens(self._tokenizer)
        self._all_token_ids = set(self.tokens.keys())

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
        logit_bias: Optional[Dict[str, int]] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(f"Unsupported decoding strategy for {self.__class__} model `{decoder.strategy}`.")

        gen_kwargs = {}
        if temperature := decoder.temperature:
            gen_kwargs["temperature"] = temperature
        if top_p := decoder.top_p:
            gen_kwargs["top_p"] = top_p
        if top_k := decoder.top_k:
            gen_kwargs["top_k"] = top_k
        if decoder.strategy == DecodingStrategy.GREEDY:
            # try to make the sampling as deterministic as possible
            # to select only the one top token
            gen_kwargs["top_p"] = (
                1 / self.vocab_size
            )  # select only n tokens to get over 1/self.vocab_size, should always be a single token
            if "top_k" in gen_kwargs:
                gen_kwargs["top_k"] = 1
        payload = {"prompt": text, "logit_bias": logit_bias, "model": self.model_name, "max_tokens": max_tokens, "logprobs": 5, **gen_kwargs}  # type: ignore

        completion_stream = await self.client.completion.create(**self.addtl_create_kwargs, **payload, stream=True)

        async for completion in completion_stream:
            yield completion

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        token_counter: Optional[TokenCount] = None,
        stream: bool = True,
    ) -> AsyncGenerator[Any, None]:
        bias = 100
        if selected_tokens and len(selected_tokens) > (self.vocab_size / 2):
            selected_tokens = self._all_token_ids - selected_tokens
            bias = -100

        if selected_tokens and len(selected_tokens) > self.max_token_selection:
            warnings.warn(
                f"Trying to mask {len(selected_tokens)} tokens which "
                f"is more than {self.max_token_selection} mask limit "
                f"of {self}. Consider stricter constraints. Will select"
                "lowest token ids up to this limit.",
            )
            selected_tokens = sorted(list(selected_tokens))[: self.max_token_selection]  # type: ignore
        logit_bias = {}
        if selected_tokens:
            for i, idx in enumerate(selected_tokens):
                logit_bias[str(idx)] = bias

        def result_handler(response):
            choice = response.choices and response.choices[0]
            content = choice and choice.delta.content or ''
            finish_reason = choice and choice.finish_reason or None
            return (
                content,  # content
                finish_reason is not None,  # complete generation
            )

        errors = []
        for retries in range(self.max_retries):
            try:
                async for chat_completion in self._generate(
                    text=text,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                    decoder=decoder,
                    timeout=timeout,
                    stream=stream,
                ):
                    content, done = result_handler(chat_completion)
                    if done:
                        break
                    if content:
                        yield (content, [None])

            except openai.APIConnectionError as exc:
                errors.append(exc)
                warnings.warn(
                    "OpenAI Completion API raised an error: \n" f"MESSAGE: {exc}\n" f"RETRYING {retries}"
                    if (retries + 1) < self.max_retries
                    else "",
                )

            if not errors:
                break
            else:
                raise AggregateException(errors)
