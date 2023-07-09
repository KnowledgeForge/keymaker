from .huggingface import Huggingface  # noqa: F401
from .openai import OpenAIChat, OpenAICompletion


def chatgpt(completion: bool = False, *args, **kwargs):
    if completion:
        return OpenAICompletion(model_name="gpt-3.5-turbo", max_total_tokens=4000, *args, **kwargs)  # type: ignore
    return OpenAIChat(model_name="gpt-3.5-turbo", max_total_tokens=4000, *args, **kwargs)  # type: ignore


def gpt4(*args, **kwargs):
    return OpenAIChat(model_name="gpt-4", max_total_tokens=8000, *args, **kwargs)  # type: ignore
