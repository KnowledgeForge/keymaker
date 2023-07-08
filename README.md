<div align="center">
  <a href="https://keymaker.headjack.ai">
    <img src="https://github.com/KnowledgeForge/keymaker/blob/main/docs/assets/images/keymaker%20logo.svg" alt="Logo" width="150">
  </a>
  <h3 align="center">KeyMaker</h3>
  <p align="center">The most powerful, flexible and extensible way to control the output of large language models.
    <br />
    <a href="https://keymaker.headjack.ai"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/KnowledgeForge/keymaker/issues">Report Bug</a>
    <br/>
    <br/>
  </p>
</div>

## Table of Contents

- [About KeyMaker](#about-keymaker) 
- [Installation](#installation)
- [Usage](#usage)
- [Basic Example](#basic-example)
- [Accessing Completions](#accessing-completions)
- [Model Options](#model-options)
  - [Huggingface (Direct)](#huggingface-direct)
  - [OpenAI](#openai)
- [Using Chat models](#using-chat-models)
  - [Mixing Chat and Non-Chat Models](#mixing-chat-and-non-chat-models)
- [Using Constraints](#using-constraints)  
  - [RegexConstraint](#regexconstraint)
  - [ParserConstraint](#parserconstraint) 
  - [OptionsConstraint](#optionsconstraint)
  - [StopsConstraint](#stopsconstraint)
- [Combining Constraints](#combining-constraints)
- [Creating Custom Models](#creating-custom-models)
- [Creating Custom Constraints](#creating-custom-constraints)  
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)
- [Copyright](#copyright)

## About KeyMaker  

KeyMaker is a Python library that provides a powerful, flexible, and extensible way 
to control the output of large language models like OpenAI GPT-3 and Hugging Face's Transformers. 
It allows you to create and apply constraints on the generated tokens, ensuring that 
the output of the model meets specific requirements or follows a desired format.

## Installation  

To install KeyMaker, simply run the following command:  

### From source:  

```sh
pip install https://github.com/KnowledgeForge/keymaker.git
```  

### From pypi:  

```sh
pip install headjack-keymaker
```    

## Usage   

### Basic Example

First, note that `Prompt`s and `Completion`s are two of the fundamental types in Keymaker.

To use KeyMaker with a language model, you need to first create a `Model` object. For example, to use KeyMaker with Hugging Face's GPT-2 model:

Some basic imports
```python
from keymaker.models import Huggingface
from keymaker import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
```

For demo purposes, we can use a local Huggingface model
```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

hf = Huggingface(model=model, tokenizer=tokenizer)
```

create a prompt using the Prompt class
```python
prompt: Prompt = Prompt("Dogs are known for their")
```

`Prompt`s are interchangeable with strings
```python
>>> prompt == "Dogs are known for their"
True
```

generate a basic completion with no constraints
`max_tokens` and `name` are optional
```python
completed_prompt: Prompt = await prompt.complete(
    model=hf, max_tokens=1, name="dog_ability"
)
```

check that the original prompt is still the same
```
>>> prompt == "Dogs are known for their"
True
```

print out the completed prompt string
```python
>>> print(completed_prompt)
Dogs are known for their ability
```

`completed_prompt.completions` is a `Completions` object
and gives access to any strings created from `.complete` calls
on its parent `Prompt`.
If the `Completion` was `name`d you can access it as an attribute
on the `.completions` with `.` syntax or `['...name...']`

```python
>>> print(completed_prompt.completions.dog_ability)
 ability
```


`completed_prompt.completions.name` is a `Completion` object
which simply stores the string completion and the start stop indices in the prompt
```python
>>> print(
    completed_prompt[
        completed_prompt.completions.dog_ability.start : completed_prompt.completions[
            "dog_ability"
        ].stop
    ]
)
 ability
```

print out the Completions object for the completed prompt
```python
>>> completed_prompt.completions
Completions([], {'dog_ability': Completion(text = ' ability', start = 24, stop = 32)})
```

### Accessing Completions

When using KeyMaker to generate text with constraints, you can name the completions to easily access them later. 
All completions are stored in the `completions` attribute of a `Prompt` object.

Here's an example of how to access both named and unnamed completions:

```python
from keymaker.models import chatgpt
from keymaker import Prompt
import openai

openai.api_key = "sk-"

chat_model = chatgpt()

prompt = Prompt("The weather is ")

# Generate an unnamed completion
constraint1 = RegexConstraint(pattern=r"sunny|rainy|cloudy")
prompt = await prompt.complete(model=chat_model, constraint=constraint1)

# Generate a named completion
constraint2 = RegexConstraint(pattern=r" and (cold|warm|hot)")
prompt = await prompt.complete(
    model=chat_model, constraint=constraint2, name="temperature"
)

print(prompt)

# Access the unnamed completion
unnamed_completion = prompt.completions[0]
print(f"Unnamed completion: {unnamed_completion}")

# Access the named completion
named_completion = prompt.completions.temperature
print(f"Named completion: {named_completion}")
```
Output:
```
The weather is sunny and warm
Unnamed completion: sunny
Named completion:  and warm
```

In the example, we create a `Prompt` object with the text "The weather is ". We then generate an unnamed completion with a `RegexConstraint` that matches the words "sunny", "rainy", or "cloudy", and a named completion with a `RegexConstraint` that matches " and " followed by "cold", "warm", or "hot". 

We access the unnamed completion by indexing the `completions` attribute of the `Prompt` object, and the named completion by using the `name` as an attribute of the `completions` attribute.

### Model Options
As it stands, the models available for use out of the box are `Huggingface` models and APIs implementing the OpenAI spec.

KeyMaker is also designed to make it as simple as possible for you to [Add Your Own Model](#creating-custom-models)

#### Huggingface (direct)
To use Huggingface models directly, simply import the `Huggingface` `Model` class:
```python
from keymaker.models import Huggingface
```

#### OpenAI
OpenAI Models can be accessed similarly:
```python
from keymaker.models import OpenAIChat, OpenAICompletion #e.g. chatgpt/gpt4, text-davinci-003 respectively
```

#### COMING SOON
There are a number of libraries that mimic the OpenAI API

#### Llama-CPP
See [Llama-Cpp-Python](https://abetlen.github.io/llama-cpp-python/#web-server)

#### Huggingface (API) via vLLM
See [vLLM](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#openai-compatible-server)

### Using Chat models

KeyMaker provides functionality for using roles with chat models. While this is optional, lack of usage could potentially impact performance.

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

Further, should you want to intermingle the usage of chat and non-chat continuations, KeyMaker provides utilities to do so:
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
```


### Using Constraints  

KeyMaker provides several out-of-the-box constraints that can be applied when completing prompts.

KeyMaker is also designed to make it as simple as possible for you to [Add Your Own Constraint](#creating-custom-constraints)  

Let's go through some of the built-in constraint types and how to use them.

#### RegexConstraint

`RegexConstraint` allows you to constrain the generated text based on a regex pattern. For example, to generate text that starts with "Hello":

```python
from keymaker.constraints import RegexConstraint

constraint = RegexConstraint(pattern=r"I (would|could) eat [0-9]{1, 2} burgers\.")

prompt = await Prompt("Hello, ").complete(model=chat_model, constraint=constraint, name="greeting")
print(prompt)
# Hello, I could eat 8 burgers.
```

This will generate text starting with "Hello, " that matches the regex pattern.

#### ParserConstraint  

`ParserConstraint` allows you to constrain the generated text based on a context-free grammar or a pre-built parser. For example, to generate text that follows a simple grammar:

```python 
from keymaker.constraints import ParserConstraint

grammar = """
start: "A" "B" "C"
"""

constraint = ParserConstraint(grammar=grammar)
```

To apply this constraint, pass it to the `complete` method: 

```python
prompt = Prompt("Start: ")
prompt = await prompt.complete(model=hf, constraint=constraint, name="grammar")
print(prompt) 
```

This will generate the text "A B C" to match the grammar.

#### OptionsConstraint  

`OptionsConstraint` allows you to constrain the generated text based on a list of string options. For example, to generate text that contains one of the following options:

```python
from keymaker.constraints import OptionsConstraint

options = {"apple", "banana", "orange"}
constraint = OptionsConstraint(options=options)
```

To apply this constraint, pass it to the `complete` method:  

```python  
prompt = Prompt("I would like a ") 
prompt = await prompt.complete(model=hf, constraint=constraint, name="fruit")
print(prompt)
```

This will generate text containing one of the options, e.g. "I would like a apple".

#### StopsConstraint  

`StopsConstraint` allows you to constrain the generated text by stopping at a specified string. For example, to generate text that stops after the word "stop":

```python
constraint = StopsConstraint(stop="!")

prompt = Prompt(
    """
%system%You are an agent that says short phrases%/system%
%user%Be very excited with your punctuation and give me a short phrase about dogs.%/user%
"""
)


prompt = await prompt.complete(model=chat_model, constraint=constraint, name="stop")
print(prompt)
# %system%You are an agent that says short phrases%/system%
# %user%Be very excited with your punctuation and give me a short phrase about dogs.%/user%
# "Dogs are absolutely pawsome!"
```

### Combining Constraints  

KeyMaker also allows you to combine multiple constraints using logical operators like `AndConstraint`, `OrConstraint`, and `NotConstraint`. 

```python
from keymaker.constraints import OrConstraint, RegexConstraint, OptionsConstraint

regex_constraint = RegexConstraint(pattern=r"peanut")
options_constraint = OptionsConstraint(options={"apple", "banana", "orange"})

combined_constraint = OrConstraint([regex_constraint, options_constraint])

prompt = Prompt("Whenever I see a basketball, it reminds me of my favorite fruit the ")

prompt = (await prompt.complete(model=chat_model, constraint=combined_constraint)) + "."

print(prompt)
#Prompt('Whenever I see a basketball, it reminds me of my favorite fruit the orange.')
```

This will generate text starting with "Hello, I have a " and containing one of the options, e.g.
"Hello, I have a apple".

### Creating Custom Models  

To create a custom model, you need to extend the `Model` class provided by KeyMaker and implement the required methods. Here's an example of creating a custom model:

```python
from keymaker.models.base import Model
from typing import AsyncGenerator, Optional, Set
from keymaker.constraints.base import Constraint
from keymaker.types import Decoder

class CustomModel(Model):
    def __init__(self, ...):  # Add any required initialization parameters
        # Initialize your custom model here
        pass  

    async def generate(
        self,
        text: str,
        max_tokens: int = 1,
        selected_tokens: Optional[Set[int]] = None,
        decoder: Optional[Decoder] = None,
        timeout: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        # Implement the logic for generating text with your custom model
        pass  

    def encode(self, text: str) -> TokenIds:
        # Implement the logic for encoding text as token ids
        pass  

    def decode(self, ids: TokenIds) -> str:
        # Implement the logic for decoding token ids as text
        pass  

    # ...
```  

You can then use your custom model with KeyMaker as you would with the built-in models:

```python
model = CustomModel(...)
prompt = Prompt("My custom model says: ")
prompt = await prompt.complete(model=model, constraint=your_constraint, name="custom_output")  
print(prompt)
```
    
### Creating Custom Constraints

To create a custom constraint, you need to extend the `Constraint` class provided by KeyMaker and implement the `constrain_tokens` method. Here's an example of creating a custom constraint:

```python
from keymaker.constraints.base import Constraint
from keymaker.types import TokenConstraint

class CustomConstraint(Constraint):
    def __init__(self, ...):  # Add any required initialization parameters
        # Initialize your custom constraint here
        pass  

    def constrain_tokens(
        self, base_text: str, completion_text: str, model: Model
    ) -> TokenConstraint:
        # Implement the logic for constraining tokens based on your custom constraint
        pass
```

You can then use your custom constraint with KeyMaker as you would with the built-in constraints:

```python
constraint = CustomConstraint(...)
prompt = Prompt("My custom constraint says: ")
prompt = await prompt.complete(model=your_model, constraint=constraint, name="custom_constraint_output")
print(prompt)
```

## Contributing

Contributions are very welcome. Simply fork the repository and open a pull request!

## Acknowledgements

Some constraints in KeyMaker are heavily derived from the work of [Matt Rickard](https://github.com/r2d4). Specifically, ReLLM and ParserLLM.

## Disclaimer

KeyMaker and its contributors bear no responsibility for any harm done by its usage either directly or indirectly including but not limited to costs incurred by using the package (KeyMaker) with LLM vendors. The package is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## Copyright
Copyright 2023- Nick Ouellet (nick@ouellet.dev)