---
categories: ["docs"]
weight: 50
title: Accessing Completions and Working with Chat Models
tags: ["how-to", "core"]
series: ["Usage"]
series_weight: 10
lightgallery: true
---
# Accessing Completions and Working with Chat Models

In this section, we will explore how to access completions and work with chat models in Keymaker.

## Accessing Completions

When you generate completions with Keymaker, you can access them through the `completions` attribute of a `Prompt` object. Here's an example of how to access both named and unnamed completions:

```python
from keymaker import Prompt
from keymaker.models import chatgpt
from keymaker.constraints import RegexConstraint

# Initialize the model
chat_model = chatgpt()

# Create a prompt
prompt = Prompt("The weather is ")

# Generate an unnamed completion
constraint1 = RegexConstraint(pattern=r"sunny|rainy|cloudy")
prompt = await prompt.complete(model=chat_model, constraint=constraint1)

# Generate a named completion
constraint2 = RegexConstraint(pattern=r" and (cold|warm|hot)")
prompt = await prompt.complete(
    model=chat_model, constraint=constraint2, name="temperature"
)

# Access the unnamed completion
unnamed_completion = prompt.completions[0]

# Access the named completion
named_completion = prompt.completions.temperature

print(prompt)
print(f"Unnamed completion: {unnamed_completion}")
print(f"Named completion: {named_completion}")
```

## Working with Chat Models

Keymaker provides functionality for using roles with chat models. This can help improve performance by providing the model with a clearer context for the conversation. Here's an example of using roles with a chat model:

```python
from keymaker import Prompt
from keymaker.models import chatgpt

# Initialize the model
chat_model = chatgpt()

# Create a prompt with roles
prompt = Prompt("""
%system%You are a helpful assistant%/system%
%user%What is the capital of France?%/user%
%assistant%The capital of France is Paris.%/assistant%
""")

# Complete the prompt with the chat model
prompt = await prompt.complete(model=chat_model, name="next_response")

# Print the completed prompt
print(prompt)
```

By using roles, you can provide a clearer context for the conversation, which may help improve the quality of the generated responses.