try:
    from .huggingface import Huggingface  # noqa: F401
except ImportError:
    pass
from .openai import OpenAIChat, OpenAICompletion

try:
    from .llama_cpp import LlamaCpp  # noqa: F401
except ImportError:
    pass

try:
    from .litellm import LiteLLM, LiteLLMChat  # noqa: F401
except ImportError:
    pass


def chatgpt(completion: bool = False, *args, **kwargs):
    max_total_tokens = kwargs.get('max_total_tokens', 4096)
    if 'max_total_tokens' in kwargs:
        del kwargs['max_total_tokens']

    if completion:
        return OpenAICompletion(model_name="gpt-3.5-turbo", max_total_tokens=max_total_tokens, *args, **kwargs)  # type: ignore
    return OpenAIChat(model_name="gpt-3.5-turbo", max_total_tokens=max_total_tokens, *args, **kwargs)  # type: ignore


def gpt4(*args, **kwargs):
    max_total_tokens = kwargs.get('max_total_tokens', 8192)
    if 'max_total_tokens' in kwargs:
        del kwargs['max_total_tokens']
    return OpenAIChat(model_name="gpt-4", max_total_tokens=max_total_tokens, *args, **kwargs)  # type: ignore
