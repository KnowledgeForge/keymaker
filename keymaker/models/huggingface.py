"""A simple huggingface-based Model implementation for basic local inference and testing"""

from typing import (
    Set,FrozenSet,
    Optional,
    AsyncGenerator,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from dataclasses import dataclass
import asyncio
import torch
from keymaker.models.base import Model
from keymaker.types import Decoder, DecodingStrategy, TokenIds, SelectedTokens, Tokens

def transformers_tokens(tokenizer: AutoTokenizer) -> Tokens:
    tokens = {
        token_id: tokenizer.decode(token_id)
        for _, token_id in tokenizer.get_vocab().items()
    }
    return tokens

@dataclass
class Huggingface(Model):
    """
    A simple huggingface-based Model implementation 
    for basic local inference and testing
    """
    model_name: Optional[str] = None
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    chunk_size: int = 64
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        )
    )

    def __post_init__(self):
        if self.model_name is None and self.model is None and self.tokenizer is None:
            raise ValueError(
                "must specify either `model_name` or both `model` and `tokenizer`"
            )
        if (self.model is not None and self.tokenizer is None) or (
            self.model is None and self.tokenizer is not None
        ):
            raise ValueError(
                "must specify either `model_name` or both `model` and `tokenizer`"
            )
        self.model = self.model or AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.tokens = transformers_tokens(self.tokenizer)
        self._completion_buffer = {}
        if self.chunk_size < 1:
            raise ValueError(f"`chunksize` must be positive, got {self.chunksize}.")

    def encode(self, text: str) -> TokenIds:
        return self.tokenizer.encode(text)

    def decode(self, ids: TokenIds) -> str:
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    def _logit_processor(self, selected_tokens: Optional[SelectedTokens] = None):
        logits_processor = []
        if selected_tokens is not None:

            def _logits_processor(input_ids, scores):
                mask = np.ones_like(scores) * -1e10
                for token_id in selected_tokens:
                    mask[:, token_id] = 0
                scores = scores + mask
                return scores

            logits_processor.append(_logits_processor)
        return logits_processor

    async def generate(
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(
                f"Unsupported decoding strategy for Huggingface model `{decoder.strategy}`."
            )
        temperature = decoder.temperature
        top_p = decoder.top_p
        addtl = {}

        if decoder.strategy == DecodingStrategy.SAMPLE:
            addtl["do_sample"] = True

        gen_kwargs = dict(temperature=temperature, top_p=top_p, **addtl)

        n_gen = 0
        prompt_token_ids = self.tokenizer.encode(text)
        while n_gen < max_tokens:
            max_new_tokens = min(self.chunk_size, max_tokens - n_gen)
            output = await asyncio.to_thread(
                self.model.generate,
                input_ids=torch.tensor(prompt_token_ids)
                .unsqueeze(0)
                .to(self.model.device),
                max_new_tokens=max_new_tokens,
                logits_processor=self._logit_processor(selected_tokens),
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )
            new_token_ids = output[0, len(prompt_token_ids) :].detach().cpu().tolist()
            prompt_token_ids += new_token_ids
            tok_str = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            text += tok_str
            n_gen += max_new_tokens
            yield tok_str