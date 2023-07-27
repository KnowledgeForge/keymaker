---
categories: ["docs"]
weight: 50
title: Advanced Usage and Customization
tags: ["how-to", "core"]
series: ["Usage"]
series_weight: 10
lightgallery: true
---
# Advanced Usage and Customization

In this section, we will explore more advanced usage of Keymaker, including working with multiple models, custom constraints, and custom models.

## Working with Multiple Models

Keymaker allows you to use multiple models in a single prompt. You can switch between models when generating completions to take advantage of their specific capabilities. Here's an example of using both a chat model and a custom model:

```python
from keymaker import Prompt
from keymaker.models import chatgpt, CustomModel
from keymaker.constraints import OptionsConstraint

# Initialize the models
chat_model = chatgpt()
custom_model = CustomModel(...)

# Create a constraint
options_constraint = OptionsConstraint(options={"apple", "banana", "orange"})

# Create a prompt
prompt = Prompt("I would like an ")

# Complete the prompt with the chat model and constraint
completed_prompt = await prompt.complete(model=chat_model, constraint=options_constraint, name="fruit")

# Update the prompt with custom model output
completed_prompt = await completed_prompt.complete(model=custom_model, name="custom_output")

# Print the completed prompt
print(completed_prompt)
```

## Creating Custom Constraints

To create a custom constraint, extend the `Constraint` class provided by Keymaker and implement the `constrain_tokens` method:

```python
from keymaker.constraints.base import Constraint
from keymaker.types import TokenConstraint

class CustomConstraint(Constraint):
    def __init__(self, ...):
        # Initialize your custom constraint here
        pass

    async def constrain_tokens(self, base_text: str, completion_text: str, model: Model) -> TokenConstraint:
        # Implement the logic for constraining tokens based on your custom constraint
        pass
```

Use your custom constraint with Keymaker as you would with built-in constraints:

```python
constraint = CustomConstraint(...)
prompt = Prompt("My custom constraint says: ")
prompt = await prompt.complete(model=your_model, constraint=constraint, name="custom_constraint_output")
print(prompt)
```

## Creating Custom Models

To create a custom model, extend the `Model` class provided by Keymaker and implement the required methods:

```python
from keymaker.models.base import Model
from typing import AsyncGenerator, Optional, Set, Dict
from keymaker.constraints.base import Constraint
from keymaker.types import Decoder

class CustomModel(Model):
    def __init__(self, ...):
        # Initialize your custom model here
        self.tokens: Dict[int, str] = ... # tokens is a required attribute
        pass

    async def generate(self, text: str, max_tokens: int, selected_tokens: Optional[Set[int]], decoder: Optional[Decoder], timeout: float) -> AsyncGenerator[str, None]:
        # Implement the logic for generating text with your custom model
        pass

    def encode(self, text: str) -> TokenIds:
        # Implement the logic for encoding text as token ids
        pass

    def decode(self, ids: TokenIds) -> str:
        # Implement the logic for decoding token ids as text
        pass
```

Use your custom model with Keymaker as you would with built-in models (here demonstrated with a single completion):

```python
model = CustomModel(...)
prompt = Prompt("My custom model says: ")
prompt = await prompt.complete(model=model, constraint=your_constraint, name="custom_output")
print(prompt)
```