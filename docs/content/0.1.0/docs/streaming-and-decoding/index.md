---
title: 'Streaming and Decoding'
date: 2019-02-11T19:30:08+10:00
draft: false
weight: 4
---
# Streaming Completions and Custom Decoding Parameters

In this section, we will explore the usage of streaming completions and custom decoding parameters in Keymaker.

## Streaming Completions

Keymaker allows you to access completion generation as it happens by passing an asynchronous function to the `complete` method. This can be useful for situations where you want to display the generated text in real-time, such as in a chat application.

Here's an example of streaming completions:

```python
from keymaker import Prompt, Completion
from keymaker.models import chatgpt
from typing import Optional

# Define a stream function
async def my_stream(completion: Optional[Completion]):
    print(completion)

# Initialize the model
chat_model = chatgpt()

# Create a prompt
prompt = Prompt("Hello, I'm a talking dog and my name is")

# Generate completions with streaming
prompt = await prompt.complete(
    model=chat_model,
    stream=my_stream,
)
```

## Custom Decoding Parameters

Keymaker allows you to set some of the decoding parameters used to sample tokens during the text generation process. You can customize the temperature, top_p, and decoding strategy.

Here's an example of using custom decoding parameters:

```python
from keymaker import Prompt, Decoder, DecodingStrategy
from keymaker.models import chatgpt

# Define custom decoding parameters
decoder = Decoder(temperature=0.7, top_p=0.95, strategy=DecodingStrategy.GREEDY)

# Initialize the model
chat_model = chatgpt()

# Create a prompt
prompt = Prompt("The weather today is ")

# Generate completions with custom decoding parameters
prompt = await prompt.complete(model=chat_model, decoder=decoder, name="weather")

# Print the completed prompt
print(prompt)
```

By customizing the decoding parameters, you can control the output of the language model to better suit your needs.