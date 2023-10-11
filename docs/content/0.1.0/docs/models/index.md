---
title: 'Models in Keymaker'
date: 2019-02-11T19:30:08+10:00
draft: false
weight: 4
summary: Various Keymaker models.
---

### Model Options
As it stands, the models available for use out of the box are `Huggingface` models and APIs implementing the OpenAI spec.

Keymaker is also designed to make it as simple as possible for you to [Add Your Own Model](#creating-custom-models)

#### Huggingface (direct)
Huggingface models are optional, and you need to install Keymaker with `pip install "headjack-keymaker[huggingface]"`, then, simply import the `Huggingface` `Model` class:
```python
from keymaker.models import Huggingface
```

#### OpenAI
OpenAI Models can be accessed out-of-the-box:
```python
from keymaker.models import OpenAIChat, OpenAICompletion #e.g. chatgpt/gpt4, text-davinci-003 respectively
```

There are aliases for common models:
```python
from keymaker.models import chatgpt, gpt4

chat_model=gpt4(...optional configurations for underlying `OpenAIChat` otherwise use defaults)
```

##### Azure OpenAI

To use the the Azure API with Keymaker is simple:

As documented in the OpenAI Python API you can set the following to your values:
```python
import openai
openai.api_type = "azure"
openai.api_key = ""
openai.api_base = "https://azureai....openai.azure.com/"
openai.api_version = "..."
```

Then, simply use the `addtl_create_kwargs` on any OpenAI based Keymaker `Model`. Here shown with chatgpt alias:
```python
model = chatgpt(addtl_create_kwargs=dict(deployment_id="gpt-35-turbo-chatgpt"))
```

#### Llama-CPP

Keymaker also provides an implementation wrapper around [Llama-Cpp-Python](https://abetlen.github.io/llama-cpp-python)

```python
from keymaker.models import LlamaCpp
from keymaker.constraints import RegexConstraint
from keymaker import Prompt

model = LlamaCpp(model_path="~/Downloads/orca-mini-v2_7b.ggmlv3.q3_K_S.bin")

constraint = RegexConstraint(r"I (eat|drink) (meat|wine)\.")
prompt = Prompt("I'm a farmer and ")

prompt = await prompt.complete(model=model, constraint=constraint)
# Prompt('I'm a farmer and I eat meat.')
```

This can be enabled by installing the optional dependencies with `pip install "headjack-keymaker[llamacpp]"`

#### OpenAI Compatible Servers
**Coming Soon - Ripe for contibution**

Keymaker is looking to make the OpenAI `Model` support other compatible APIs. Simply pass a compatible tokenizer and go!

##### Llama-CPP
See [Llama-Cpp-Python](https://abetlen.github.io/llama-cpp-python/#web-server)

##### Huggingface (API) via vLLM
**Cuda Only**
See [vLLM](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#openai-compatible-server)

### Using Chat models

Keymaker provides functionality for using roles with chat models. While this is optional, lack of usage could potentially impact performance.

Chat models (e.g. `OpenAIChat` or the aliases `chatgpt`, `gpt`) have the following default attributes (which can vary should you [Add Your Own Model](#creating-custom-models))
```python
    role_tag_start = "%"
    role_tag_end = "%"
    default_role = "assistant"
    allowed_roles = ("system", "user", "assistant")
```

This affects the way your prompt will be seen by the chat model. For example:
```python
prompt = Prompt(
    """
%system%You are an agent that says short phrases%/system%
%user%Be very excited with your punctuation and give me a short phrase about dogs.%/user%
"Dogs are absolutely pawsome!"
"""
)
```
would be seen by the chat model as:
```python
[{'role': 'system', 'content': 'You are an agent that says short phrases'},
 {'role': 'user',
  'content': 'Be very excited with your punctuation and give me a short phrase about dogs.'},
 {'role': 'assistant', 'content': '"Dogs are absolutely pawsome!"'}]
```

#### Mixing Chat and Non-Chat Models

Further, should you want to intermingle the usage of chat and non-chat continuations, Keymaker provides utilities to do so:
```python
from keymaker.utils import strip_tags

prompt = Prompt(
    """
%system%You are an agent that says short phrases%/system%
%user%Be very excited with your punctuation and give me a short phrase about dogs.%/user%
"Dogs are absolutely pawsome!"
"""
)

regular_prompt = strip_tags(prompt, roles_seps = {'system': '', 'user': 'User: ', 'assistant': 'Assistant: '},)
>>> regular_prompt
```
Result:
```python
Prompt('You are an agent that says short phrases
User: Be very excited with your punctuation and give me a short phrase about dogs.
Assistant: "Dogs are absolutely pawsome!"')