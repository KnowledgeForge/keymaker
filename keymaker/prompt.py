"""The fundamental components of Keymaker Prompts and Completions"""
import asyncio
import warnings
from typing import Optional

from keymaker.constraints.base import Constraint
from keymaker.models.base import Model
from keymaker.types import Decoder


class Completion(str):
    """A completion string from a prompt

    Args:
        text (str): the generated string
        start (int): the start index of the completion in the prompt it came from
        stop (int): the stop index of the completion in the prompt it came from
        chunk (bool): whether the completion is part of a larger whole

    Returns:
        Completion (str)
    """

    def __new__(cls, text: str, start: int, stop: int, name: Optional[str] = None, chunk: bool = False):
        if isinstance(text, Completion):
            return text
        obj = str.__new__(cls, text)
        obj.start = start  # type: ignore
        obj.stop = stop  # type: ignore
        obj.chunk = chunk  # type: ignore
        obj.name = name #type: ignore
        return obj

    def __repr__(self) -> str:
        return f"Completion(text = '{self}', start = {self.start}, stop = {self.stop})"  # type: ignore


class Completions:
    def __init__(self):
        self._completions = []
        self._named_completions = {}

    def __repr__(self) -> str:
        return f"Completions({self._completions}, {self._named_completions})"

    def add(self, completion, name=None):
        if name is not None:
            self._named_completions[name] = completion
        else:
            self._completions.append(completion)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._completions[key]
        else:
            return self._named_completions[key]

    def __getattr__(self, name):
        if name in self._named_completions:
            return self._named_completions[name]
        else:
            raise AttributeError(f"'Completions' object has no attribute '{name}'")

    def __or__(self, other):
        if isinstance(other, Completions):
            combined = Completions()
            combined._completions = self._completions + other._completions
            combined._named_completions = {
                **self._named_completions,
                **other._named_completions,
            }
            return combined
        else:
            raise TypeError(f"unsupported operand type(s) for |: 'Completions' and '{type(other).__name__}'")


class Prompt(str):
    """A Prompt is a piece of text a model can generate off of

    Args:
        prompt (str): the string representing the current completion of the Prompt

    Returns:
        Prompt (str)
    """

    def __new__(cls, prompt: str, completions: Optional[Completions] = None):
        if isinstance(prompt, Completion):
            return prompt
        obj = str.__new__(cls, prompt)
        obj.prompt = prompt  # type: ignore
        obj.completions = completions or Completions()  # type: ignore
        return obj

    def __repr__(self):
        return f"Prompt('{self.prompt}')"  # type: ignore

    def __str__(self):
        return self.prompt

    def __add__(self, other):
        if isinstance(other, str):
            return Prompt(self.prompt + other, self.completions)  # type: ignore
        elif isinstance(other, Prompt):
            return Prompt(self.prompt + other.prompt, self.completions | other.completions)  # type: ignore
        else:
            raise TypeError(f"Cannot concatenate Prompt object with object of type {type(other)}")

    def __radd__(self, other):
        if isinstance(other, str):
            return Prompt(other + self.prompt, self.completions)  # type: ignore
        elif isinstance(other, Prompt):
            return Prompt(other.prompt + self.prompt, self.completions | other.completions)  # type: ignore
        else:
            raise TypeError(f"Cannot concatenate object of type {type(other)} with Prompt object")

    def token_length(self, model: Model) -> int:
        return len(model.encode(self.prompt))  # type: ignore

    async def complete(
        self,
        model: Model,
        constraint: Optional[Constraint] = None,
        name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        decoder: Optional[Decoder] = None,
        stream_queue: Optional[asyncio.Queue] = None,
        timeout: float = 10.0,
        truncate: bool = False,
    ):
        text = self.prompt  # type: ignore
        prompt_tokens = model.encode(self.prompt)  # type: ignore
        token_limit = min(max_tokens or 1_000_000_000, model.max_total_tokens - len(prompt_tokens))
        if truncate and (len(prompt_tokens) + (max_tokens or 0)) >= model.max_total_tokens:
            warnings.warn(
                f"Prompt plus `max_tokens` more than model `max_total_tokens` of {model.max_total_tokens}."
                "Truncating from right.",
            )
            text = model.decode(prompt_tokens[-(model.max_total_tokens - token_limit) :])  # noqa: E203

        if max_tokens is not None and max_tokens > token_limit:
            warnings.warn(
                f"Requested `max_tokens` of {max_tokens} "
                f"greater than remaining token limit {token_limit} "
                f"from model {str(model)[:10]}...) which has "
                f"`max_total_tokens` {model.max_total_tokens}. "
                f"will limit `max_tokens` to {token_limit}.",
            )
        
        if constraint is None:
            generated = ""
            async for tok in model.generate(text, max_tokens=token_limit, decoder=decoder, timeout=timeout):
                if stream_queue:
                    await stream_queue.put(Completion(
                            tok,
                            len(self.prompt)+len(generated),  # type: ignore
                            len(self.prompt) + len(generated)+len(tok),  # type: ignore
                            name,
                            True
                        )
                    )
                generated += tok
            if stream_queue:
                await stream_queue.put(None)
            ret = self + generated
            ret.completions.add(# type: ignore
                        Completion(
                            generated,
                            len(self.prompt),  # type: ignore
                            len(self.prompt) + len(generated),  # type: ignore
                            name
                        ),
                        name,
                    )
            return ret

        token_count = 0
        partial_completion = ""
        prompt_plus_completion = text[:]
        while token_count < token_limit:
            selected_token_ids = constraint.constrain_tokens(text, partial_completion, model)
            if selected_token_ids:
                selected_token_ids = None if len(selected_token_ids) > model.vocab_size else selected_token_ids
            if isinstance(selected_token_ids, set) and len(selected_token_ids) == 0:
                warnings.warn(f"Empty token mask encountered with Constraint `{constraint}`. Ending completion.")
                break

            if isinstance(selected_token_ids, str):
                partial_completion = selected_token_ids
                break
            if isinstance(selected_token_ids, set) and len(selected_token_ids) == 1:
                generation = model.decode(list(selected_token_ids))
            else:
                generation = await model.sample(
                    prompt_plus_completion,
                    selected_tokens=selected_token_ids,
                    decoder=decoder,
                    timeout=timeout,
                )
            if model.encode(generation)[-1] == model.eos_token_id:
                break
            if stream_queue:
                await stream_queue.put(Completion(
                        tok,
                        len(self.prompt)+len(partial_completion),  # type: ignore
                        len(self.prompt) + len(partial_completion)+len(tok),  # type: ignore
                        name,
                        True
                    )
                )
            partial_completion += generation
            prompt_plus_completion = self.prompt + partial_completion  # type: ignore
            token_count += 1
        if stream_queue:
            await stream_queue.put(None)

        ret = self + partial_completion
        ret.completions.add(
            Completion(
                partial_completion,
                len(self.prompt),  # type: ignore
                len(self.prompt) + len(partial_completion),  # type: ignore
                name
            ),
            name,
        )

        return ret
