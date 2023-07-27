---
title: 'Getting Started'
date: 2019-02-11T19:30:08+10:00
draft: false
weight: 4
summary: How to get started with Keymaker.
---

# Getting Started with Keymaker

Keymaker is a Python library that provides a powerful and flexible way to control the output of large language models. This guide will walk you through installing Keymaker and using it to generate some completions including:
  - with constraints
  - different models
  - streaming

## Installation

Install Keymaker with:

```shell
pip install headjack-keymaker
```

For HuggingFace support:

```shell
pip install headjack-keymaker[huggingface] 
```

For LlamaCpp support:

```shell
pip install headjack-keymaker[llamacpp]
```

For everything:

```shell
pip install headjack-keymaker[all]
```

## Basic Usage

Import Keymaker:

```python
from keymaker import Prompt
```

Create a prompt:

```python 
prompt = Prompt("Dogs are known for being man's")
```

Import a model - here we will use Huggingface, but Keymaker supports OpenAI chat completion, OpenAI completion, LlamaCPP, and OpenAI API compatible models.

```python
from keymaker.models import Huggingface

model = Huggingface("gpt2")
```

Make a single one-off completion of the prompt with a model:

```python
completed = await prompt.complete(model=model, max_tokens=3)
```

The completed prompt text is now available:

```python
print(completed)
# Dogs are known for being man's best friend.
```

## Formatting and Constraints

Beyond one-off completions, Keymaker supports an extremely flexible formatted completion and of course constraints.

Import a constraint and `CompletionConfig`:

```python
from keymaker import CompletionConfig, Prompt
from keymaker.models import Huggingface, chatgpt
from keymaker.constraints import RegexConstraint, OptionsConstraint
```

Set up the model:

```python
model = Huggingface("gpt2")
```

Define a function to print the stream:

```python
async def print_stream(completion):
    if completion:
        print(completion)
```

Provide a default model and stream which will be used if we do not specify otherwise:

```python
prompt = Prompt(
    "My name is {} and I am {age} years old. I like {hobbies}",
    model, 
    stream=print_stream,
)
```

Define possible hobbies:

```python
possible_hobbies = {"Reading", "Sports", "Gaming", "Traveling", "Cooking"}
```

Set the OpenAI API key for using chatgpt:

```python
import openai

openai.api_key = ""

chat_model = chatgpt()
```

Here, we define a function to generate unique hobbies:

```python
def hobbies(prompt: Prompt):
    """Generate Unique Hobbies"""
    while True:
        if prompt.completions["hobbies"] is None:
            hobbies_selected = set()
        elif isinstance(prompt.completions.hobbies, list):
            yield ", "
            hobbies_selected = set(prompt.completions.hobbies)
        else:
            yield ", "
            hobbies_selected = {prompt.completions.hobbies}
        if len(hobbies_selected) > 1:
            yield "and Soccer."
            break
        hobbies_left = possible_hobbies - hobbies_selected
        yield CompletionConfig(
            model=chat_model, constraint=OptionsConstraint(hobbies_left)
        )
```

Keymaker supports the following as format parameters:
- `Stringable`: Anything that can be cast to a `str`.
- `CompletionConfig`: A single parameterized `Completion`.
- `Callable[[Prompt], Stringable | CompletionConfig]`: **Dynamically change the prompt completion by returning a SINGLE static `Stringable` or request a paramterized `Completion`** with a function that takes the current prompt at that point of completion.
- `Callable[[Prompt], Generator[Stringable | CompletionConfig]]`: **Dynamically change the prompt completion by returning as many static `Stringable` or paramterized `Completion`s** with a `Generator`


Fill the prompt:

```python
filled = await prompt.format(
    "Alice",
    age=CompletionConfig(
        constraint=RegexConstraint(r"[1-9]{1,2}", terminate_on_match=True), map_fn=int
    ),
    hobbies=hobbies,
)
```

The streamed output will be:

```shell
My name is 
Alice
 and I am 
33
 years old. I like 
Reading
, 
Travel
, 
and Soccer.
```

You can access the age completion:

```python
age = filled.completions.age
```

Finally, print the filled prompt and the age:

```python
print(filled)
print(age, type(age.value))
# My name is Alice and I am 33 years old. I like Reading, Traveling, and Soccer.
# 33 <class 'int'>
```

Congratulations! You've completed the guide to getting started with Keymaker.
