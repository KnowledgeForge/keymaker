"""A simple huggingface-based Model implementation for basic local inference and testing"""

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, FrozenSet, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from keymaker.models.base import Model
from keymaker.types import Decoder, DecodingStrategy, SelectedTokens, TokenIds, Tokens
from keymaker.utils.general import TokenCount


def transformers_tokens(tokenizer: AutoTokenizer) -> Tokens:
    tokens = {token_id: tokenizer.decode(token_id) for _, token_id in tokenizer.get_vocab().items()}
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
    supported_decodings: FrozenSet[DecodingStrategy] = frozenset(  # type: ignore
        (
            DecodingStrategy.GREEDY,
            DecodingStrategy.SAMPLE,
        ),
    )

    def __post_init__(self):
        if self.model_name is None and self.model is None and self.tokenizer is None:
            raise ValueError("must specify either `model_name` or both `model` and `tokenizer`")
        if (self.model is not None and self.tokenizer is None) or (self.model is None and self.tokenizer is not None):
            raise ValueError("must specify either `model_name` or both `model` and `tokenizer`")
        self.model = self.model or AutoModelForCausalLM.from_pretrained(self.model_name)

        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(self.model_name)
        self.tokens = transformers_tokens(self.tokenizer)

    def encode(self, text: str) -> TokenIds:
        return self.tokenizer.encode(text, add_special_tokens=True)  # type: ignore

    def decode(self, ids: TokenIds) -> str:
        return self.tokenizer.decode(ids)  # type: ignore

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id  # type: ignore

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id  # type: ignore

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

    async def generate(  # type: ignore
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[SelectedTokens] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
        token_counter: Optional[TokenCount] = None,
    ) -> AsyncGenerator[Tuple[str, List[float]], None]:
        decoder = decoder or Decoder()
        if decoder.strategy not in self.supported_decodings:
            raise ValueError(f"Unsupported decoding strategy for Huggingface model `{decoder.strategy}`.")
        gen_kwargs = {}
        if temperature := decoder.temperature:
            gen_kwargs["temperature"] = temperature
        if top_p := decoder.top_p:
            gen_kwargs["top_p"] = top_p
        if top_k := decoder.top_k:
            gen_kwargs["top_k"] = top_k

        if decoder.strategy == DecodingStrategy.SAMPLE:
            gen_kwargs["do_sample"] = True

        n_gen = 0
        prompt_token_ids = self.tokenizer.encode(text)  # type: ignore
        while n_gen < max_tokens:
            max_new_tokens = max_tokens - n_gen
            output = await asyncio.to_thread(
                self.model.generate,  # type: ignore
                input_ids=torch.tensor(prompt_token_ids).unsqueeze(0).to(self.model.device),  # type: ignore
                max_new_tokens=max_new_tokens,
                logits_processor=self._logit_processor(selected_tokens),
                pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )
            new_token_ids = output.sequences[0, len(prompt_token_ids) :].detach().cpu().tolist()  # noqa: E203

            logprobs = np.log(
                [
                    logits.detach().cpu().squeeze().softmax(0)[tok_id].item()
                    for tok_id, logits in zip(new_token_ids, output.scores)
                ],
            ).tolist()
            prompt_token_ids += new_token_ids
            tok_str = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)  # type: ignore
            text += tok_str
            n_gen += max_new_tokens
            yield tok_str, logprobs

    async def probs(  # type: ignore
        self,
        text: str,
    ) -> List[Tuple[str, float]]:
        prompt_token_ids = self.tokenizer.encode(text)  # type: ignore
        output = await asyncio.to_thread(
            self.model.generate,  # type: ignore
            input_ids=torch.tensor(prompt_token_ids).unsqueeze(0).to(self.model.device),  # type: ignore
            max_new_tokens=1,
            pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
            return_dict_in_generate=True,
            output_scores=True,
        )
        return list(zip(self.tokens.values(), output.scores[0].detach().cpu().squeeze().softmax(0).tolist()))
