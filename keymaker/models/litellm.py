"""Models to use with litellm"""
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, FrozenSet, List, Optional, Tuple

from litellm import Router, decode, encode
from litellm.utils import _select_tokenizer

from keymaker.models.base import ChatModel, Model
from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens
from keymaker.utils.chat import split_tags
from keymaker.utils.general import TokenCount


@lru_cache(10)
def litellm_tokens(type, tokenizer) -> Tokens:

    if 'hugging' in type:
        return {token_id: tokenizer.decode(token_id) for _, token_id in tokenizer.get_vocab().items()}

    if 'openai' in type:
        vocab_len = len(tokenizer.token_byte_values())
        tokens = {i: tokenizer.decode([i]) for i in range(vocab_len - 1)}
        for i in range(vocab_len, tokenizer.max_token_value):
            tokens[i] = f"<|special_{i}|>"
        return tokens

    raise ValueError(f"Could not handle getting the vocabulary for `{(type, tokenizer)}`.")


@dataclass
class LiteLLM(Model):
    """Class representing all LiteLLM model usage.

    client (litellm.Router): The router object for accessing LiteLLM functionality (see: https://docs.litellm.ai/docs/routing)
    model_name (str): must be in the `client` (i.e. the model_list of the litellm.Router must contain an element of `model_name`)
    supported_decodings (FrozenSet[DecodingStrategy]): A set of supported decoding strategies for generating text.
    role_tag_start (str): The starting tag used to specify role information in the input text.
    role_tag_end (str): The ending tag used to specify role information in the input text.
    default_role (str): The default role to assign to text segments that do not have an explicitly specified role.
    allowed_roles (FrozenSet[str]): A set of allowed roles for text segments.
    max_token_selection (int): The maximum number of tokens that can be selected during token masking.
    max_total_tokens (int): The maximum number of tokens the model can handle (includes prompt and completion).
    sample_chunk_size (int): The number of tokens generated by the generate method when using the sample decoding strategy.
    addtl_create_kwargs (Dict[str, Any]): Additional keyword arguments to be passed to completion call
        (Note: some of these parameters are defined explicitly in this class as necessary. See: https://docs.litellm.ai/docs/completion/input#input-params-1).
    """

    client: Router
    model_name: str
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )
    role_tag_start: str = "%"
    role_tag_end: str = "%"
    default_role: str = "user"
    allowed_roles: FrozenSet[str] = frozenset(("system", "user", "assistant"))
    max_token_selection: int = 300
    max_total_tokens: int = 4096
    sample_chunk_size: int = 12  # tokens generated by `generate` on behalf of `sample`
    addtl_create_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._tokenizer = _select_tokenizer(self.model_name)
        self.tokens = litellm_tokens(**self._tokenizer)
        self._all_token_ids = set(self.tokens.keys())

    def encode(self, text: str) -> TokenIds:
        return encode(model=self.model_name, text=text)

    def decode(self, ids: TokenIds) -> str:
        return decode(model=self.model_name, tokens=ids)

    @property
    def eos_token_id(self) -> int:
        try:
            return self._tokenizer.encode_single_token("<|endofprompt|>", allowed_special={"<|endofprompt|>"})
        except Exception:
            raise ValueError(f"Could not access eos token with LiteLLM model `{self.model_name}`.")

    @property
    def bos_token_id(self) -> int:
        try:
            return self._tokenizer.encode_single_token("<|endofprompt|>", allowed_special={"<|endofprompt|>"})
        except Exception:
            raise ValueError(f"Could not access bos token with LiteLLM model `{self.model_name}`.")

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

        # if messages[-1]["role"] != self.default_role:
        #     messages.append({"role": self.default_role, "content": " "})

        if token_counter is not None:
            tot_text = "".join(m["content"] for m in messages)
            token_counter.add_prompt_tokens(len(self.encode(tot_text)))

        gen_kwargs = gen_kwargs or  {}
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
            **gen_kwargs, # type: ignore
            "messages": messages,
            "logit_bias": logit_bias,
            "model": self.model_name,
            "max_tokens": max_tokens,
        }

        completion_stream = await self.client.acompletion(**self.addtl_create_kwargs, **payload, stream=stream)
        
        if not stream:
            class Choice:
                finish_reason = completion_stream.choices[0].finish_reason
                delta = completion_stream.choices[0].message
            class Chunk:
                choices = [Choice]
            yield Chunk
        else:
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

        first_iter = False
        async for chat_completion in self._generate(
            text=text,
            max_tokens=max_tokens,
            logit_bias=logit_bias,
            decoder=decoder,
            timeout=timeout,
            token_counter=token_counter,
            gen_kwargs=gen_kwargs,
            stream=stream,
        ):
            
            if first_iter and token_counter is not None and token_counter.model_str != chat_completion.model:
                token_counter.set_model_str(chat_completion.model)

            content, done = result_handler(chat_completion)
            if content:
                yield (content, [None] if stream else [None]*len(self.encode(content)))
            if done:
                break

            first_iter = False


@dataclass
class LiteLLMChat(ChatModel, LiteLLM):
    default_role: str = "assistant"
