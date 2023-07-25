"""The fundamental components of Keymaker Prompts and Completions"""
import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generator, List, Optional, Union

from keymaker.constraints.base import Constraint
from keymaker.models.base import ChatModel, Model
from keymaker.types import Decoder, FormatArg, Stringable
from keymaker.utils.formatted_completion import format_parser
from keymaker.utils.general import add_logprob, anoop, exp, noop


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

    def __new__(
        cls,
        value: Stringable,
        start: int,
        stop: Optional[int] = None,
        name: Optional[str] = None,
        chunk: bool = False,
        score: Optional[float] = None,
    ):
        if isinstance(value, Completion):
            return value
        obj = str.__new__(cls, str(value))
        return obj

    def __init__(
        self,
        value: Stringable,
        start: int,
        stop: Optional[int] = None,
        name: Optional[str] = None,
        chunk: bool = False,
        score: Optional[float] = None,
    ):
        self.value = value
        self.start = start
        self.stop = stop or (start + len(self))
        self.chunk = chunk
        self.name = name
        self.score = score

    def __repr__(self) -> str:
        return f"Completion(text='{self}', value=`{self.value}`, start={self.start}, stop={self.stop}, name={self.name}, chunk={self.chunk}, score={self.score})"

    def map(self, fn: Callable[["Completion"], Stringable]) -> "Completion":
        mapped = fn(self)
        return Completion(
            value=mapped,
            start=self.start,
            stop=self.start + len(str(mapped)),
            chunk=self.chunk,
            name=self.name,
            score=self.score,
        )


class Completions:
    def __init__(self):
        self._completions: List[Completion] = []
        self._named_completions: Dict[str, Union[List[Completion], Completion]] = {}

    def __repr__(self) -> str:
        return f"Completions({self._completions}, {self._named_completions})"

    def add(self, completion, name=None):
        if name is not None:
            if name in self._named_completions:
                extant = self._named_completions[name]
                if isinstance(extant, list):
                    extant.append(completion)
                else:
                    self._named_completions[name] = [extant, completion]
            else:
                self._named_completions[name] = completion
        else:
            self._completions.append(completion)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._completions[key]
        else:
            return self._named_completions.get(key)

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


@dataclass
class CompletionConfig:
    model: Optional[Model] = None
    constraint: Optional[Constraint] = None
    name: Optional[str] = None
    max_tokens: Optional[int] = None
    decoder: Optional[Decoder] = None
    stream: Optional[Callable[[Optional['Completion']], Awaitable[Any]]] = None
    map_fn: Callable[[Completion], Stringable] = noop
    timeout: float = 10.0
    truncate: bool = False
    try_first: Optional[bool] = None

    def __or__(self, other):
        if isinstance(other, CompletionConfig):
            combined = CompletionConfig()
            combined.model = self.model or other.model
            combined.constraint = self.constraint or other.constraint
            combined.name = self.name or other.name
            combined.max_tokens = self.max_tokens or other.max_tokens
            combined.decoder = self.decoder or other.decoder
            combined.stream = self.stream or other.stream
            combined.map_fn = self.map_fn or other.map_fn
            combined.timeout = self.timeout or other.timeout
            combined.truncate = self.truncate or other.truncate
            combined.try_first = self.try_first or other.try_first
            return combined
        else:
            raise TypeError(f"unsupported operand type(s) for |: 'CompletionConfig' and '{type(other).__name__}'")


class Prompt(str):
    """A Prompt is a piece of text a model can generate off of

    Args:
        prompt (str): the string representing the current completion of the Prompt

    Returns:
        Prompt (str)
    """

    def __new__(cls, prompt: str, completions: Optional[Completions] = None, *default_args, **default_kwargs):
        if isinstance(prompt, Prompt):
            return prompt
        obj = str.__new__(cls, prompt)
        return obj

    def __init__(self, prompt: str, *default_args, completions: Optional[Completions] = None, **default_kwargs):
        self.prompt = prompt
        self.completions = completions or Completions()
        if 'completion_config' in default_kwargs:
            self._default_completion_config = default_kwargs['completion_config']
        else:
            self._default_completion_config = CompletionConfig(*default_args, **default_kwargs)

    def __repr__(self):
        return f"Prompt('{self.prompt}')"

    def __str__(self):
        return self.prompt

    def __add__(self, other):
        if isinstance(other, Prompt):
            return Prompt(
                self.prompt + other.prompt,
                completions=self.completions | other.completions,
                completion_config=self._default_completion_config | other._default_completion_config,
            )
        if isinstance(other, str):
            return Prompt(self.prompt + other, completions=self.completions, completion_config=self._default_completion_config)
        raise TypeError(f"Cannot concatenate Prompt with object of type {type(other)}.")

    def __radd__(self, other):
        if isinstance(other, Prompt):
            return Prompt(
                other.prompt + self.prompt,
                completions=self.completions | other.completions,
                completion_config=self._default_completion_config | other._default_completion_config,
            )
        if isinstance(other, str):
            return Prompt(other + self.prompt, completions=self.completions, completion_config=self._default_completion_config)
        raise TypeError(f"Cannot concatenate object of type {type(other)} with Prompt.")

    def __getitem__(self, key):
        return Prompt(super().__getitem__(key), completions=self.completions, completion_config=self._default_completion_config)

    def token_length(self, model: Model) -> int:
        return len(model.encode(self.prompt))

    async def format(self, *args: FormatArg, **kwargs: FormatArg) -> "Prompt":  # type: ignore
        formattings = format_parser(self)
        prompt = self[:0]
        unnamed_index = 0
        default_stream = self._default_completion_config.stream or anoop
        for part in formattings:
            if isinstance(part, str):
                completion = Completion(part, len(prompt), None, None, False, None)
                await default_stream(completion)
                prompt += completion
                continue
            name = part.name
            if name is None:
                config = args[unnamed_index]
                unnamed_index += 1
            else:
                config = kwargs[name]
            addition = None
            if isinstance(config, CompletionConfig):
                config.name = name if name is not None else config.name
                prompt = await prompt.complete(completion_config=config)
            elif callable(config):
                config = config(prompt)  # type: ignore
                if isinstance(config, Generator):
                    for config_chunk in config:
                        if isinstance(config_chunk, CompletionConfig):
                            config_chunk.name = name if name is not None else config_chunk.name
                            prompt = await prompt.complete(completion_config=config_chunk)
                        else:
                            completion = Completion(config_chunk, len(prompt), None, None, False, None)
                            await default_stream(completion)
                            prompt += completion
                else:
                    if isinstance(config, CompletionConfig):
                        config.name = name if name is not None else config.name
                        prompt = await prompt.complete(completion_config=config)
                    else:
                        addition = config
            else:
                addition = config  # type: ignore
            if addition is not None:
                completion = Completion(addition, len(prompt), None, None, False, None)
                await default_stream(completion)
                prompt += completion
        return prompt

    async def complete(self, *completion_args, **completion_kwargs) -> "Prompt":
        if 'completion_config' in completion_kwargs:
            config = completion_kwargs['completion_config']
        else:
            config = CompletionConfig(*completion_args, **completion_kwargs)
        config = config | self._default_completion_config

        model = config.model
        constraint = config.constraint
        name = config.name
        max_tokens = config.max_tokens
        decoder = config.decoder
        stream = config.stream
        map_fn = config.map_fn
        timeout = config.timeout
        truncate = config.truncate
        try_first = config.try_first

        if model is None:
            raise ValueError("A model is required for completion")

        ret = self[:]
        if try_first is None:
            try_first = isinstance(model, ChatModel)
        text = self.prompt
        prompt_tokens = model.encode(self.prompt)
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

            async for tok, logprob in model.generate(text, max_tokens=token_limit, decoder=decoder, timeout=timeout):
                logprobs_sum = add_logprob(logprobs_sum, *logprob)  # type: ignore
                if stream:
                    await stream(
                        Completion(
                            value=tok,
                            start=len(self.prompt) + len(generated),
                            stop=len(self.prompt) + len(generated) + len(tok),
                            name=name,
                            chunk=True,
                            score=exp(add_logprob(0, *logprob)),
                        ),
                    )

                generated += tok
            if stream:
                await stream(None)

            completion = Completion(
                generated,
                len(self.prompt),
                len(self.prompt) + len(generated),
                name,
                False,
                exp(logprobs_sum),
            ).map(map_fn)

            ret.completions.add(completion, name)

            ret = self + completion
            return ret

        token_count = 0
        partial_completion = ""
        buffer_tokens: List[str] = []
        buffer_logprobs: List[float] = []
        token = ""
        free_attempt = try_first

        async def send_stream(value: Stringable, chunk: bool = False, logprobs: Optional[float] = None):
            if stream:
                await stream(
                    Completion(
                        value=value,
                        start=len(text) + len(partial_completion),
                        stop=None,
                        name=name,
                        chunk=chunk,
                        score=exp(logprobs),
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
                text + partial_completion,
                selected_tokens=None if free_attempt else selected_token_ids,
                decoder=decoder,
                timeout=timeout,
            )

            if not generated_tokens:
                break

            token = generated_tokens[0]

            # token generated according to constraint is already good
            if not free_attempt:
                partial_completion += token
                buffer_tokens = generated_tokens[1:]
                buffer_logprobs = logprobs[1:]
                logprobs_sum = add_logprob(logprobs_sum, logprobs[0])  # type: ignore
                await send_stream(token, True, add_logprob(0, logprobs[0]))
                token_count += 1
            else:  # tried but need to validate token
                token_id = model.encode(token)[0]
                if selected_token_ids is None or token_id in selected_token_ids:
                    partial_completion += token
                    buffer_tokens = generated_tokens[1:]
                    buffer_logprobs = logprobs[1:]
                    logprobs_sum = add_logprob(logprobs_sum, logprobs[0])  # type:ignore
                    free_attempt = True  # will be allowed to try again next round
                    await send_stream(token, True, add_logprob(0, logprobs[0]))
                    token_count += 1
                else:
                    free_attempt = False

            # if there is a buffer, validate it
            end_completion = False
            for i, (token, bl) in enumerate(zip(buffer_tokens, buffer_logprobs)):
                selected_token_ids = await constraint.constrain_tokens(text, partial_completion + token, model)
                if isinstance(selected_token_ids, str):
                    partial_completion = selected_token_ids
                    token_count = len(model.encode(partial_completion))
                    end_completion = True
                    break
                elif selected_token_ids != set():
                    await send_stream(token, True, add_logprob(0, bl))
                    partial_completion += token
                    token_count += 1
                else:
                    break

            buffer_tokens = []
            buffer_logprobs = []

            if end_completion:
                break

        if stream:
            await stream(None)

        completion = Completion(
            value=partial_completion,
            start=len(self.prompt),
            stop=len(self.prompt) + len(partial_completion),
            name=name,
            chunk=False,
            score=exp(logprobs_sum),
        ).map(map_fn)

        ret.completions.add(completion, name)

        ret = self + completion
        return ret
