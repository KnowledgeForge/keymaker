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

To use KeyMaker with a language model, you need to first create a `Model` object. For example, to use KeyMaker with Hugging Face's GPT-2 model:

```python
from keymaker.models import Huggingface
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

hf = Huggingface(model=model, tokenizer=tokenizer, chunk_size=3)
```

To generate text without constraints, simply use the `generate` method: 

```python
async for t in hf.generate("Hello, world! ", 10):
    print(t)
```

This will generate 10 tokens of text starting with "Hello, world! ".

### Accessing Named and Unnamed Completions

When using KeyMaker to generate text with constraints, you can name the completions to easily access them later. Named completions are stored in the `completions` attribute of a `Prompt` object, while unnamed completions are stored in a list.

Here's an example of how to access both named and unnamed completions:

```python
from keymaker import Prompt
from keymaker.constraints import RegexConstraint

prompt = Prompt("The weather is ")

# Generate an unnamed completion
constraint1 = RegexConstraint(pattern=r"sunny|rainy|cloudy")
prompt = await prompt.complete(model=your_model, constraint=constraint1)

# Generate a named completion
constraint2 = RegexConstraint(pattern=r" and (cold|warm|hot)")
prompt = await prompt.complete(model=your_model, constraint=constraint2, name="temperature")

# Access the unnamed completion
unnamed_completion = prompt.completions[0]
print(f"Unnamed completion: {unnamed_completion}")

# Access the named completion
named_completion = prompt.completions.temperature
print(f"Named completion: {named_completion}")
```

In the example, we create a `Prompt` object with the text "The weather is ". We then generate an unnamed completion with a `RegexConstraint` that matches the words "sunny", "rainy", or "cloudy", and a named completion with a `RegexConstraint` that matches " and " followed by "cold", "warm", or "hot". 

We access the unnamed completion by indexing the `completions` attribute of the `Prompt` object, and the named completion by using the `name` as an attribute of the `completions` attribute.

### Using Constraints  

KeyMaker provides several constraints that can be applied to the generated text. Let's go through some of the constraint types and how to use them.

#### RegexConstraint  

`RegexConstraint` allows you to constrain the generated text based on a regex pattern. For example, to generate text that starts with "Hello":

```python
from keymaker.constraints import RegexConstraint

constraint = RegexConstraint(pattern=r"Hello.*")
```

To apply this constraint, pass it to the `complete` method:

```python
prompt = Prompt("Hello, ")
prompt = await prompt.complete(model=hf, constraint=constraint, name="greeting")
print(prompt)
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
from keymaker.constraints import StopsConstraint

stop = "stop"
constraint = StopsConstraint(stop=stop, include=True)
```

To apply this constraint, pass it to the `complete` method:  

```python
prompt = Prompt("Keep going until you ")
prompt = await prompt.complete(model=hf, constraint=constraint, name="stop")
print(prompt)
```

This will generate text stopping at the word "stop", e.g. 
"Keep going until you stop".

### Combining Constraints  

KeyMaker also allows you to combine multiple constraints using logical operators like `AndConstraint`, `OrConstraint`, and `NotConstraint`. 

For example, to generate text that starts with "Hello" and contains one of the specified options:
```python
from keymaker.constraints import AndConstraint, RegexConstraint, OptionsConstraint

regex_constraint = RegexConstraint(pattern=r"Hello.*")
options_constraint = OptionsConstraint(options={"apple", "banana", "orange"})

combined_constraint = AndConstraint([regex_constraint, options_constraint])
```

To apply the combined constraint, pass it to the `complete` method:

```python
prompt = Prompt("Hello, I have a ")
prompt = await prompt.complete(model=hf, constraint=combined_constraint, name="fruit") 
print(prompt)
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

    # Optionally, implement other required properties and methods
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

Some constraints in KeyMaker are heavily derived from the work of [Matt Rickard ↗](https://github.com/r2d4). Specifically, ReLLM and ParserLLM.

t