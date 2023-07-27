---
categories: ["docs"]
weight: 50
title: Formatting and Control FLow
tags: ["how-to", "core"]
series: ["Usage"]
series_weight: 10
lightgallery: true
---

# Advanced Prompt Formatting and Control Flow

In this section, we will explore advanced prompt formatting and plain control flow with Keymaker.

## Advanced Prompt Formatting

Keymaker provides powerful and flexible ways to format prompts. You can use the `.format()` method to generate completions based on the order of arguments, named arguments, or even dynamic arguments using callables and generators. Here's an example:

```python
from keymaker import Prompt, Completion, CompletionConfig
from keymaker.models import chatgpt
from keymaker.constraints import RegexConstraint, StopsConstraint
from datetime import datetime

# Initialize the model
chat_model = chatgpt()

# Create a formatted prompt
prompt = Prompt("""
Time: {time}
User: {user_msg}
Assistant: Hello, {}{punctuation}
User: Can you write me a poem about a superhero named pandaman being a friend to {}?
Assistant:{poem}
User: What is 10+5?
Assistant: The answer is 10+5={math}

The final answer is {fin}!

User: Countdown from 5 to 0.
Assistant: 5, 4, {countdown}
""")

# Define a countdown function
def countdown(prompt):
    while True:
        count = prompt.completions["countdown"]
        count = count[-1] if isinstance(count, list) else count
        if count is None or int(count.strip(", ")) > 0:
            yield CompletionConfig(
                chat_model,
                constraint=RegexConstraint("[0-9]"),
                map_fn=lambda s: f"{s}, ",
            )
        else:
            break

# Fill in the prompt with completions
filled_in = await prompt.format(
    CompletionConfig(constraint=OptionsConstraint({"Sam", "Nick"})),
    lambda p: p.completions[0],
    punctuation="!",
    user_msg="Hi, my name is Nick.",
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    poem=CompletionConfig(
        chat_model,
        max_tokens=250,
        constraint=StopsConstraint("User|Assistant", include=False),
    ),
    math=CompletionConfig(
        chat_model,
        constraint=RegexConstraint("[0-9]+", terminate_on_match=False),
        map_fn=lambda s: f"{s}",
    ),
    fin=lambda p: CompletionConfig(
        chat_model,
        constraint=RegexConstraint(rf"{p.completions.math}|16"),
    ),
    countdown=countdown,
)

# Print the completed prompt
print(filled_in)
```

## Control Flow in Prompts

Of course, beyond the `format` methodology, Keymaker provides the more basic primitive `complete`. Here's an example of using control flow with `complete`:

```python
from keymaker import Prompt, CompletionConfig
from keymaker.models import chatgpt
from keymaker.constraints import RegexConstraint

# Initialize the model
chat_model = chatgpt()

# Create a prompt
prompt = Prompt("I'm a farmer and ")

# Complete the prompt with a constraint
constraint = RegexConstraint(r"I (eat|drink) (meat|wine)\.")
prompt = await prompt.complete(model=chat_model, constraint=constraint, name="farmer_diet")

# Modify the prompt based on the completion
if "meat" in prompt:
    prompt = Prompt(prompt.completions.farmer_diet) + " This means that"

# Continue with completions
prompt = await prompt.complete(...)
```

By using control flow with `complete` or with callables and `format`, you can generate prompts that adapt to the context and provide more relevant and accurate completions.