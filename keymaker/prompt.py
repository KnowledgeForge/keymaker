"""The fundamental components of Keymaker Prompts and Completions"""
import warnings
from typing import Any, Awaitable, Callable, List, Optional
from keymaker.constraints.base import Constraint
from keymaker.models.base import ChatModel, Model
from keymaker.types import Decoder
import math

class Completion(str):
    """A completion string generated from a prompt.

    Args:
        text (str): The generated completion string.
        start (int): The start index of the completion in the original prompt. 
        stop (int): The stop index of the completion in the original prompt.
        name (Optional[str]): A name for the completion.
        chunk (bool): Whether the completion is part of a larger chunk.
        score (Optional[float]): The score for the completion from the model.

    Returns:
        Completion: A string subclass with additional completion metadata.
    """

    def __new__(cls, text: str, start: int, stop: int, name: Optional[str] = None, chunk: bool = False, score: Optional[float] = None):
        if isinstance(text, Completion):
            return text
        obj = str.__new__(cls, text)
        obj.start = start  # type: ignore
        obj.stop = stop  # type: ignore
        obj.chunk = chunk  # type: ignore
        obj.name = name  # type: ignore
        obj.score = score # type: ignore
        return obj

    def __repr__(self) -> str:
        return f"Completion(text='{self}', start={self.start}, stop={self.stop}, name={self.name}, chunk={self.chunk}, score={self.score})"


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


def add_logprob(logprobs_sum: Optional[float], *logprobs: Optional[float])->Optional[float]:
    if logprobs_sum is None or None in logprobs:
        return None
    return logprobs_sum + sum(logprobs)

def exp(x: Optional[float])->Optional[float]:
    if x is None:
        return None
    return math.exp(x)

class Prompt(str):
    """A Prompt is a piece of text a model can generate off of

    Args:
        prompt (str): the string representing the current completion of the Prompt

    Returns:
        Prompt (str)
    """

    def __new__(cls, prompt: str, completions: Optional[Completions] = None):
        if isinstance(prompt, Prompt):
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
        stream: Optional[Callable[[Optional[Completion]], Awaitable[Any]]] = None,
        timeout: float = 10.0,
        truncate: bool = False,
        try_first: Optional[bool] = None
    ):
        if try_first is None:
            try_first = isinstance(model, ChatModel)
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
        logprobs_sum = 0
        if constraint is None:
            generated = ""
            async for tok, logprob in model.generate(text, max_tokens=token_limit, decoder=decoder, timeout=timeout):  # type: ignore
                logprobs_sum = add_logprob(logprobs_sum, *logprob)
                if stream:
                    await stream(
                        Completion(
                            tok,
                            len(self.prompt) + len(generated),  # type: ignore
                            len(self.prompt) + len(generated) + len(tok),  # type: ignore
                            name,
                            True,
                            exp(logprobs_sum)
                        ),
                    )
                
                generated += tok
            if stream:
                await stream(None)
            ret = self + generated
            ret.completions.add(  # type: ignore
                Completion(
                    generated,
                    len(self.prompt),  # type: ignore
                    len(self.prompt) + len(generated),  # type: ignore
                    name,
                    False,
                    exp(logprobs_sum)
                ),
                name,
            )
            return ret

        token_count = 0
        partial_completion = ""
        buffer_tokens: List[str] = []
        buffer_logprobs: List[float] = []
        token = ""
        free_attempt = try_first
        async def send_stream(text: str, chunk: bool = False):
            if stream:
                await stream(
                    Completion(
                        text,
                        len(text) + len(partial_completion),  # type: ignore
                        len(text) + len(partial_completion) + len(text),  # type: ignore
                        name,
                        chunk,
                        exp(logprobs_sum)
                    ),
                )
        while token_count < token_limit:
            selected_token_ids = await constraint.constrain_tokens(text, partial_completion, model)
            
            if selected_token_ids:
                # if the selected tokens have the same number than the vocab size, there's no real restriction
                selected_token_ids = None if len(selected_token_ids) >= model.vocab_size else selected_token_ids
                
            if isinstance(selected_token_ids, set) and len(selected_token_ids) == 0:
                warnings.warn(f"Empty token mask encountered with Constraint `{constraint}`. Ending completion.")
                break

            # constraints return strings when they are finished 
            # to explictly suggest the completion
            if isinstance(selected_token_ids, str):
                partial_completion = selected_token_ids
                break
            generated_tokens, logprobs = await model.sample(
                text+partial_completion,
                selected_tokens=None if free_attempt else selected_token_ids,
                decoder=decoder,
                timeout=timeout,
            )

            if not generated_tokens:
                break
            
            token = generated_tokens[0]

            # token generated according to constraint is already good
            if not free_attempt:
                partial_completion+=token
                buffer_tokens = generated_tokens[1:]
                buffer_logprobs = logprobs[1:]
                logprobs_sum=add_logprob(logprobs_sum, logprobs[0])
                await send_stream(token, True)
                token_count+=1
            else: # tried but need to validate token
                token_id = model.encode(token)[0]
                if selected_token_ids is None or token_id in selected_token_ids:
                    partial_completion+=token
                    buffer_tokens = generated_tokens[1:]
                    buffer_logprobs = logprobs[1:]
                    logprobs_sum=add_logprob(logprobs_sum, logprobs[0])
                    free_attempt = True # will be allowed to try again next round
                    await send_stream(token, True)
                    token_count+=1
                else:
                    free_attempt = False

            # if there is a buffer, validate it
            end_completion = False
            for i, (token, bl) in enumerate(zip(buffer_tokens, buffer_logprobs)):
                selected_token_ids = await constraint.constrain_tokens(text, partial_completion+token, model)
                if isinstance(selected_token_ids, str):
                    partial_completion = selected_token_ids
                    token_count = len(model.encode(partial_completion))
                    end_completion=True
                    break
                elif selected_token_ids != set():
                    await send_stream(token, True)
                    partial_completion += token
                    token_count += 1
                else:
                    break
                

                           
            buffer_tokens=[]
            buffer_logprobs=[]

            if end_completion:
                break

        if stream:
            await stream(None)

        ret = self + partial_completion
        ret.completions.add(  # type: ignore
            Completion(
                partial_completion,
                len(self.prompt),  # type: ignore
                len(self.prompt) + len(partial_completion),  # type: ignore
                name,
                False, 
                exp(logprobs_sum)
            ),
            name,
        )

        return ret
