"""A model for use with an in-memory Llama Cpp"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, FrozenSet, Optional, Dict, Any
import numpy as np
from keymaker.models.base import Model
from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens

try: 
    from llama_cpp import Llama
    import llama_cpp
    from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
except ImportError:
    raise ImportError(f"llama-cpp-python is an optional dependency and is not installed.")

def try_decode(tok_bytes: bytes)->str:
    try:
        return tok_bytes.decode('utf-8')
    except:
        return str(tok_bytes)[2:-1]

def llama_tokens(llm: "Llama") -> Tokens:#type: ignore
    vocab_len = llm.n_vocab()#type: ignore
    decoded_already = set()
    tokens = {}
    for i in range(vocab_len - 1):
        decoded = try_decode(llm.detokenize([i]))#type: ignore
        if decoded and (decoded not in decoded_already):
            tokens[i] = decoded
            decoded_already.add(decoded)
    return tokens

@dataclass
class LlamaCpp(Model):
    """
    A LlamaCpp-based Model implementation
    """

    model_path: Optional[str] = None
    model: Optional["Llama"] = None#type: ignore
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )
    cache_type: Optional[str] = None
    cache_size: int = 2 << 30
    llama_kwargs: Optional[Dict[str, Any]] = None

        
    def __post_init__(self):
        if self.model_path is None and self.model is None:
            raise ValueError("must specify either `model_name` or `model`")
        if self.model_path is not None and self.model is not None:
            raise ValueError("must specify either `model_name` or `model` not both")
        self.model = self.model or Llama(model_path=self.model_path, **(self.llama_kwargs or {}))#type: ignore
        self.tokens = llama_tokens(self.model)
        if self.cache_type is not None:
            if self.cache_type == "disk":
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=self.cache_size)# type: ignore
            else:
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=self.cache_size)# type: ignore
            self.model.set_cache(cache)
            
    def encode(self, text: str) -> TokenIds:
        return self.model.tokenize(text.encode('utf-8'), add_bos = False)  # type: ignore

    def decode(self, ids: TokenIds) -> str:
        return try_decode(self.model.detokenize(ids))  # type: ignore

    @property
    def eos_token_id(self) -> int:
        return self.model.token_eos()  # type: ignore

    @property
    def bos_token_id(self) -> int:
        return self.model.token_bos()  # type: ignore

    def _logit_processor(self, selected_tokens: Optional[SelectedTokens] = None):
        logits_processor = None
        if selected_tokens is not None:

            def _logits_processor(input_ids, scores):
                mask = np.ones_like(scores) * -1e10
                for token_id in selected_tokens:
                    mask[token_id] = 0
                scores = scores + mask
                return scores

            logits_processor=_logits_processor
        return logits_processor

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(f"Unsupported decoding strategy for LlamaCpp model `{decoder.strategy}`.")

        gen_kwargs = {'prompt': text, 'stream': True, 'max_tokens': max_tokens}
        if temperature:= decoder.temperature:
            gen_kwargs['temp'] = temperature
        if top_p:= decoder.top_p:
            gen_kwargs['top_p'] = top_p
        if top_k:= decoder.top_k:
            gen_kwargs['top_k'] = top_k

        if logits_processor:=self._logit_processor(selected_tokens):
            gen_kwargs['logits_processor']=logits_processor

        token_generator = await run_in_threadpool(self.model, **gen_kwargs)  # type: ignore
        async for chunk in iterate_in_threadpool(token_generator):
            yield chunk['choices'][0]['text']