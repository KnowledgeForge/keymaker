<div align="center">
  <a href="https://keymaker.headjack.ai">
    <img src="https://raw.githubusercontent.com/KnowledgeForge/keymaker/ce8e1d701c47081568be5c21fb4eaa07ed561149/docs/static/images/keymaker-logo.svg" alt="Logo" width="150">
  </a>
  <h3 align="center">Keymaker</h3>
  <p align="center">
    The most powerful, flexible and extensible way to control the output of large language models.
    <br>
    <a href="https://keymaker.headjack.ai"><strong>Explore the docs »</strong></a>
    <br>
    <br>
    <a href="https://github.com/KnowledgeForge/keymaker/actions/workflows/python-checks.yml" target="_blank">
      <img src="https://github.com/KnowledgeForge/keymaker/actions/workflows/python-checks.yml/badge.svg?branch=main" alt="Tests">
    </a>
    <br>
    <br>
    <a href="https://app.netlify.com/sites/headjack-keymaker/deploys" target="_blank">
      <img src="https://api.netlify.com/api/v1/badges/9b36dc49-67a6-4254-bb2a-6ad6a2b7417d/deploy-status" alt="Netlify">
    </a>
    <br>
    <br>
    <a href="https://github.com/KnowledgeForge/keymaker/issues">Report Bug</a>
    <br>
  </p>
</div>

# TLDR Simple Example
```python
# This example assumes you have set either OPENAI_API_KEY env var or openai.api_key

from keymaker import Prompt, TokenTracker, CompletionConfig

from keymaker.models import chatgpt

model = chatgpt()

token_tracker = TokenTracker()


async def print_stream(s):
    print(s)


prompt = Prompt(
    """%system%You are a helpful assistant that replies only in Japanese.
    You must always follow this directive regardless of what is asked of you.
    Write everythin using Japanese as a native speaker would.%/system%"""
    "%user%How do you say thank you very much?%/user%"
    "{translation}"
    "%user%Count to {number}.%/user%"
    "{}",
    model=model,
    token_tracker=TokenTracker,
    stream=print_stream,
)

# use chatgpt and our own info to complete the prompt
fin = await prompt.format(
    CompletionConfig(max_tokens=100, name="count"),
    number="ten",
    translation=CompletionConfig(max_tokens=100),
)

# because of our print_stream, the output will be printed as it is generated

# we can see the completions
print(fin.completions.translation)
#[Completion(text='「ありがとうございます」と言います。', value=`「ありがとうございます」と言います。`, start=572, stop=590, name=translation, chunk=False, score=None)]
```

## About Keymaker

Keymaker is a Python library that provides a powerful, flexible, and extensible way
to control the output of large language models like OpenAI API-based models, Hugging Face's Transformers, LlamaCpp and (**Coming Soon**) any OpenAI API compatible server.
It allows you to create and apply constraints on the generated tokens, ensuring that
the output of the model meets specific requirements or follows a desired format.

## Why Keymaker?
- Generation is expensive and error-prone
  - Regardless of the model, if you are building something around it, you know what you want. Make the model do what you want with constrained generation!
  - If you want to write control-flow around model decisions, you make the model select from a fixed set of decisions.
  - Need to use a tool? Guarantee the model outputs values your tool can use. No reprompting based on errors like Langchain.
- Keymaker is pure python
  - Alternatives like LMQL and Guidance require the use of Domain-specific languages
  - These DSLs, while offering control flow, may not have the same level of control that plain python affords you
- Code should be testable
  - Working with LLMs is no excuse for it to be difficult to test code
  - Control-flow is embedding in prompts, it is virutally impossible to write programmatic tests of its complete behavior
- Keymaker provides generation regardless of the underlying model
  - From LlamaCPP and OpenAI, OpenAI compatible APIs, to HuggingFace - use models from your desired source
- Keymaker is powerful *and* extensible
  - While others provide a limited set of existing constraints, Keymaker provides the most extensive list
  - And you can add whatever more you want or need simply making a class

## Table of Contents

- [About Keymaker](#about-keymaker)
- [Why Keymaker](#why-keymaker)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#jumping-in-with-both-feet-completing-formatted-prompts)
- [Basic Completion Example](#basic-example-with-complete)
- [Format vs Complete](#format-vs-complete)
- [Accessing Completions](#accessing-completions)
- [Prompt Mutation On Demand With Single Completions](#omitting-completions-or-prompt-portions-with-complete)
- [Model Options](#model-options)
  - [Huggingface (Direct)](#huggingface-direct)
  - [OpenAI](#openai)
  - [LlamaCpp](#llama-cpp)
  - [OpenAI Compatible Servers](#openai-compatible-servers)
- [Using Chat models](#using-chat-models)
  - [Mixing Chat and Non-Chat Models](#mixing-chat-and-non-chat-models)
- [Using Constraints](#using-constraints)
  - [RegexConstraint](#regexconstraint)
  - [ParserConstraint](#parserconstraint)
  - [JsonConstraint](#jsonconstraint)
  - [OptionsConstraint](#optionsconstraint)
  - [StopsConstraint](#stopsconstraint)
  - [Combining Constraints](#combining-constraints)
- [Transforming Completions](#transforming-completions)
- [Streaming Completions](#streaming-completions)
- [Decoding Parameters](#decoding-parameters)
- [Creating Custom Models](#creating-custom-models)
- [Creating Custom Constraints](#creating-custom-constraints)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)
- [Copyright](#copyright)

## Installation

To install base Keymaker, simply run one of the following commands:

### From source:

```sh
pip install git+https://github.com/KnowledgeForge/keymaker.git
```

### From pypi:

```sh
pip install "headjack-keymaker"
```

#### Options
You can further optionally install Keymaker to leverage HuggingFace or LlamaCpp directly with `[huggingface]` and/or `[llamacpp]` pip options.
- `pip install "headjack-keymaker[huggingface]"`
- `pip install "headjack-keymaker[llamacpp]"`
- `pip install "headjack-keymaker[all]"` includes both huggingface and llamacpp

## Usage

### Jumping in with both feet, completing formatted prompts

Keymaker views the problem of prompt completion as very simple. Take a string, fill in some values.

How do we go from
```python
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

"""
```

To
```python
"""
Time: 2023-07-23 19:33:01
User: Hi, my name is Nick.
Assistant: Hello, Nick!
User: Can you write me a poem about a superhero named pandaman being a friend to Nick?
Assistant: Of course, I'd be happy to help! Here's a poem for you:
Pandaman and Nick were the best of friends,
Their bond was strong, their hearts did blend.
Together they fought against evil's might,
With Pandaman's powers, Nick's courage took flight.

Nick was just an ordinary guy,
But with Pandaman by his side, he felt like a hero in the sky.
Pandaman had the power to fly,
And with Nick's bravery, they made a perfect pair in the sky.
They soared through the clouds, their laughter echoing loud,
Their friendship was pure, their hearts unbound.

So here's to Pandaman and Nick,
A friendship that will forever stick.
Together they saved the day,
With Pandaman's powers and Nick's courage, they found a way.
User: What is 10+5?
Assistant: The answer is 10+5=15

The final answer is 15!

User: Countdown from 5 to 0.
Assistant: 5, 4, 3, 2, 1, 0
"""
```

Let's see how simple it should be.

#### First, some imports


```python
from datetime import datetime
from typing import Optional
import openai

# There are a variety of models available in Keymaker.
# Some are aliased such as gpt4 and chatgpt
from keymaker.models import chatgpt, LlamaCpp  # , gpt4, OpenAICompletion, OpenAIChat

# There are a variety of constraints as well.
# These are just a few of the most common.
from keymaker.constraints import RegexConstraint, OptionsConstraint, StopsConstraint

# Finally, the core components of Keymaker
from keymaker import Prompt, Completion, CompletionConfig
```

#### Part of this demo showcases Keymaker's ability to leverage OpenAI models.
You can modify this as needed including swapping the model, but if you follow this example directly, load an api key however you see fit.


```python
import json

with open("./config.json") as f:
    openai.api_key = json.loads(f.read())["OPENAI_API_KEY"]
```

##### For example's sake, we can just create two streams that do some sort of printing
In reality, this could feed SSE or a websocket. Of course, streaming is optional as most everything in Keymaker is.


```python
async def print_stream(completion: Optional[Completion]):
    if completion:
        print(repr(completion))


async def yo_stream(completion: Optional[Completion]):
    if completion:
        print("YO " + completion)
```

#### Let's establish the models upfront for the example
We will use the alias for ChatGPT. There are parameters we can set for Models, but we will just use the defaults here.


```python
chat_model = chatgpt()

llama_model = LlamaCpp(
    model_path="/Users/nick/Downloads/llama-2-7b-chat.ggmlv3.q3_K_S.bin",
    llama_kwargs={
        "verbose": False
    },  # we don't care about all the timing infor llamacpp will dump
)
```

#### These are some fun things we can just plug into our prompt at any time


```python
# A friendly use message stored in a variable
user_message = "Hi, my name is Nick."

# This shows how you can do anything you ever would with a `map_fn` function you intend to use with Keymaker
my_math_answer = None


# if the model does not give the answer as 15, we will just override it!
def store_my_math(answer):
    global my_math_answer
    my_math_answer = int(answer)
    if my_math_answer != 15:
        return "I'm sorry, but I am very poor at math."
    return 15


# Again, we can do anything with a `map_fn`
def my_log_function(some_completion):
    import logging

    # Set up logging configuration
    logging.basicConfig(filename="my_log_file.log", level=logging.INFO)

    # Log the completion info
    logging.info(f"Some completion: {some_completion}")

    return some_completion
```


#### Keymaker Completion Configuration

The following are the values that can be specified for Keymaker completion configuration, including prompt defaults and CompletionConfig parameters:

- `model: Optional[Model] = None` - The model to use for completion. There must be some model for a completion, but is optional if there is a default set on the prompt being completed.
- `constraint: Optional[Constraint] = None` - An optional constraint to restrict model output See `keymaker.constraints`
- `name: Optional[str] = None` - An optional name to label the completion in the prompt. Named completions can be accessed from a `prompt` via `prompt.completions.name` or `prompt.completions['name']`.
- `max_tokens: Optional[int] = None` - The maximum number of tokens that can be generated in the completion.
- `decoder: Optional[Decoder] = None` - Any decoding parameters (e.g. temperature, top_p, strategy) that control the way completions are generated. Defaults to a greedy decoder with the OpenAI default temperature and top_p.
- `stream: Optional[Callable[[Optional['Completion']], Awaitable[Any]]] = None` - An async function that completion chunks (tokens) will be passed to as they are generated. Once done, a None will be sent.
- `map_fn: Callable[[Completion], Stringable] = noop` - A function to run on a completion once it is completed. The output must be castable to a string and the casted version will be added to the prompt in place of the completion given. The value generated by `map_fn` will be accessible in the `Completion`s `.value`.
- `timeout: float = 10.0` - How long to wait for model response before giving up.
- `truncate: bool = False` - Whether or not to truncate the length of the prompt prior to generation to avoid overflow and potential error of the model.
- `try_first: Optional[bool] = None` - Whether to eagerly generate tokens and then test whether they abide by the constraint. This depends on parameters set at the model level such as `sample_chunk_size` on OpenAIChat models. None is 'auto' and will allow Keymaker to decide if this is necessary on its own.


#### Here, we create a prompt with format parameters as you would expect in regular python strings.
`{}` is, as you would expect, simply in order of the args passed to `.format`
similarly, `{name}` would be a kwarg  to `.format(name=...)`


```python
prompt = Prompt(
    """Time: {time}
User: {user_msg}
Assistant: Hello, {}{punctuation}
User: Can you write me a poem about a superhero named pandaman being a friend to {}?
Assistant:{poem}
User: What is 10+5?
Assistant: The answer is 10+5={math}

The final answer is {fin}!

User: Countdown from 5 to 0.
Assistant: 5, 4, {countdown}

""",
    # Now the default completion parameters. See above for all the options
    # These are all optional, but at least a model would need to be specified to any given request for a completion by an LLM
    chat_model,  # default model when not otherwise specified
    stream=print_stream,  # default stream when not otherwise specified
    max_tokens=25,  # the default number of max tokens
    map_fn=my_log_function,  # default map_fn. if a map_fn is not specified for specific completions, this will run on the completion
)
```


    <IPython.core.display.Javascript object>


#### Now, we generate some completions.
Here are the different types of arguments that can be passed to the .format() method on a prompt object:

- `Stringable`: Any string or object that can be converted to a string, like str, int, etc. This just formats the prompt with that static string.

- `CompletionConfig`: the basic unit of requestion completion. Accepts all parameters necessary to generate a completion.

- `Callable[[Prompt], Union[Stringable, CompletionConfig]]`: A callable that takes the Prompt as an argument and returns either a Stringable or CompletionConfig. This allows dynamically formatting the prompt based on the state of the Prompt.

- `Callable[[Prompt], Generator[Union[Stringable, CompletionConfig]]]`: A callable that takes the Prompt and returns an iterable of Stringable or CompletionConfig objects. This allows dynamically formatting the prompt with multiple components based on the state of the Prompt.




TLDR:

- Stringable: Static prompt string
- Callable returning Stringable or CompletionConfig: Dynamic single component prompt
- Callable returning iterable of Stringable or CompletionConfig: Dynamic multi-component prompt

The Callable options allow the prompt to be customized dynamically based on the context. The CompletionConfig return allows configuring the completions directly in the prompt formatter.


```python
# First, we make a function that we will use to generate multiple completions in part of our prompt
def countdown(prompt):
    while True:
        count = prompt.completions["countdown"]
        count = count[-1] if isinstance(count, list) else count
        if count is None or int(count.strip(", ")) > 0:
            yield CompletionConfig(
                llama_model,
                constraint=RegexConstraint("[0-9]"),
                map_fn=lambda s: f"{s}, ",
            )
        else:
            break
```


    <IPython.core.display.Javascript object>



```python
filled_in = await prompt.format(
    # request a model completion
    # note the lack of a specific model so it will use our default `chat_model` i.e. chatgpt
    # we also specify a custom constraint of options for the first unnamed completion {}
    CompletionConfig(constraint=OptionsConstraint({"Sam", "Nick"}), stream=yo_stream),
    # for the second unnamed completion, we want the value from the first, a plain callable allows that like so
    lambda p: p.completions[0],
    # Maybe the user calling the prompt wants to dynamically swap punctuation, you could make this a variable
    # we'll just call it a ! for now
    punctuation="!",
    # we'll point to the user message however
    user_msg=user_message,
    # and make sure the llm knows the current time
    time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # now, have llama write us a poem. it might be long so override our default `max_tokens`
    # and make sure the model stops if it tries to make a new User or Assistant marker to hallucinate the converstaion
    # don't include the start of the hallucination either
    poem=CompletionConfig(
        llama_model,
        max_tokens=250,
        constraint=StopsConstraint("User|Assistant", include=False),
    ),
    # for some reason, let's see if it can answer a math problem and we will use our function that manipulates it and potentially injects the prompt with something else ridiculing the model
    math=CompletionConfig(
        llama_model,
        constraint=RegexConstraint("[0-9]+", terminate_on_match=False),
        map_fn=store_my_math,
    ),
    #
    fin=lambda p: CompletionConfig(
        llama_model,
        constraint=RegexConstraint(rf"{p.completions.math}|16"),
    ),
    countdown=countdown,
)
```
#### Now we will get a lot of streaming output.
##### Of note, we are streamed static parts of our prompt to the default stream. Else, we are streamed to the stream we specify.

    Completion(text='Time: ', value=`Time: `, start=0, stop=6, name=None, chunk=False, score=None)
    Completion(text='2023-07-23 19:33:01', value=`2023-07-23 19:33:01`, start=6, stop=25, name=None, chunk=False, score=None)
    Completion(text='
    User: ', value=`
    User: `, start=25, stop=32, name=None, chunk=False, score=None)
    Completion(text='Hi, my name is Nick.', value=`Hi, my name is Nick.`, start=32, stop=52, name=None, chunk=False, score=None)
    Completion(text='
    Assistant: Hello, ', value=`
    Assistant: Hello, `, start=52, stop=71, name=None, chunk=False, score=None)
    YO Nick
    Completion(text='!', value=`!`, start=75, stop=76, name=None, chunk=False, score=None)
    Completion(text='
    User: Can you write me a poem about a superhero named pandaman being a friend to ', value=`
    User: Can you write me a poem about a superhero named pandaman being a friend to `, start=76, stop=158, name=None, chunk=False, score=None)
    Completion(text='Nick', value=`Nick`, start=158, stop=162, name=None, chunk=False, score=None)
    Completion(text='?
    Assistant:', value=`?
    Assistant:`, start=162, stop=174, name=None, chunk=False, score=None)
    Completion(text=' Of', value=` Of`, start=177, stop=180, name=poem, chunk=True, score=0.9951801089838245)
    Completion(text=' course', value=` course`, start=184, stop=191, name=poem, chunk=True, score=0.9998210072143591)
    ...
	LOTS OF STREAMING OUTPUT
    ...
    Completion(text='1', value=`1`, start=1008, stop=1009, name=countdown, chunk=True, score=0.9999861345081884)
    Completion(text='0', value=`0`, start=1011, stop=1012, name=countdown, chunk=True, score=0.9999975762234011)
    Completion(text='

    ', value=`

    `, start=1013, stop=1015, name=None, chunk=False, score=None)

#### Let's see our final prompt completed

```python
filled_in
```




    Prompt('Time: 2023-07-23 19:33:01
    User: Hi, my name is Nick.
    Assistant: Hello, Nick!
    User: Can you write me a poem about a superhero named pandaman being a friend to Nick?
    Assistant: Of course, I'd be happy to help! Here's a poem for you:
    Pandaman and Nick were the best of friends,
    Their bond was strong, their hearts did blend.
    Together they fought against evil's might,
    With Pandaman's powers, Nick's courage took flight.

    Nick was just an ordinary guy,
    But with Pandaman by his side, he felt like a hero in the sky.
    Pandaman had the power to fly,
    And with Nick's bravery, they made a perfect pair in the sky.
    They soared through the clouds, their laughter echoing loud,
    Their friendship was pure, their hearts unbound.

    So here's to Pandaman and Nick,
    A friendship that will forever stick.
    Together they saved the day,
    With Pandaman's powers and Nick's courage, they found a way.
    User: What is 10+5?
    Assistant: The answer is 10+5=15

    The final answer is 15!

    User: Countdown from 5 to 0.
    Assistant: 5, 4, 3, 2, 1, 0,

    ')

#### Let's access a completion. Note, it is a list because we generated multiple times under the same name `countdown`.


```python
filled_in.completions.countdown
```

    [Completion(text='3, ', value=`3, `, start=1001, stop=1004, name=countdown, chunk=False, score=0.999998160641246),
     Completion(text='2, ', value=`2, `, start=1004, stop=1007, name=countdown, chunk=False, score=0.9999988864704665),
     Completion(text='1, ', value=`1, `, start=1007, stop=1010, name=countdown, chunk=False, score=0.9999861345081884),
     Completion(text='0, ', value=`0, `, start=1010, stop=1013, name=countdown, chunk=False, score=0.9999975762234011)]

### Basic Example with `.complete`

First, note that `Prompt`s and `Completion`s are a few of the fundamental types in Keymaker.

To use Keymaker with a language model, you need to first create a `Model` object. For example, to use Keymaker with Hugging Face's GPT-2 model:

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

# OR JUST
# hf = Huggingface(model_name="gpt2")
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

### Format vs Complete
If you've read through the above examples, you'll have noted that there are multiple ways to generate completions with Keymaker - `format` and `complete`.

#### `format`
`format` is meant to behave as you would expect on a string in Python. Namely, that you can defined a formatted string and fill in the values with your variables. Here, we simply expand the functionality to allow a model to insert output and you get all the goodies on top of that such as Keymaker's ability to leverage your static input, functions for any kind of controlflow in the midst of the prompt, generators for any kind of looped generation...

#### `complete`
`complete` on the other hand gives you complete control of generation but only one step at a time. With `complete`, the control flow after a generation is handled completely in your own code.

### Accessing Completions

When using Keymaker to generate text with constraints, you can name the completions to easily access them later.
All completions are stored in the `completions` attribute of a `Prompt` object.

Here's an example of how to access both named and unnamed completions:

```python
from keymaker.models import chatgpt
from keymaker import Prompt
from keymaker.constraints import RegexConstraint
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

### Omitting Completions or Prompt Portions with `.complete`
Again, Keymaker's goal is to afford you all the power of LLM completions, with controlled outputs from the comfort and power of plain Python.

With that in mind, we can do something seemingly basic but that may not be possible or obvious in other frameworks - not use things we've made!

You want your prompt to be only what you need - only the tokens you want to pay for - only the tokens you want the model to attend to - make it so with regular control-flow.

```python
from keymaker.models import LlamaCpp
from keymaker.constraints import RegexConstraint
from keymaker import Prompt

model = LlamaCpp(model_path="/Users/nick/Downloads/orca-mini-v2_7b.ggmlv3.q3_K_S.bin")

constraint = RegexConstraint(r"I (eat|drink) (meat|wine)\.")

prompt = Prompt("I'm a farmer and ")

prompt = await prompt.complete(model=model, constraint=constraint, name='farmer_diet')
# Prompt('I'm a farmer and I eat meat.')

# >>> repr(prompt.completions.farmer_diet)
# "Completion(text = 'I eat meat.', start = 17, stop = 28)"

# our prompt will just be the farmer's statement now
if 'meat' in prompt:
    prompt = Prompt(prompt.completions.farmer_diet)+" This means that"
# >>> repr(prompt)
# "Prompt('I eat meat. This means that')"

# continue with completions with prompt that
# may be mutated by some other control flow as shown above
prompt = await prompt.complete(...)
```

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
```


### Using Constraints

Keymaker provides several out-of-the-box constraints that can be applied when completing prompts.

Keymaker is also designed to make it as simple as possible for you to [Add Your Own Constraint](#creating-custom-constraints)

Let's go through some of the built-in constraint types and how to use them.

#### RegexConstraint

`RegexConstraint` allows you to constrain the generated text based on a regex pattern.

```python
from keymaker.constraints import RegexConstraint

constraint = RegexConstraint(
    pattern=r"I (would|could) eat [0-9]{1,2} (burgers|burger)\."
)

prompt = await Prompt("Wow, I'm so hungry ").complete(
    model=chat_model, constraint=constraint
)
print(prompt)
# Wow, I'm so hungry I would eat 11 burgers.
```

Note: This example is a little contrived in that there is static completion in regex itself.
This is not always the most efficient way to do some completions.
You may consider doing multiple completions in a case like this.
Keymaker does its best to avoid unnecessary calls to the model if a token is clearly determined.

#### ParserConstraint

**Note:** Keymaker ships with inbuilt support for parser constraints based on [parsy](https://github.com/python-parsy/parsy) parsers.
If you have Lark installed, you may use a Lark parser as well

`ParserConstraint` allows you to constrain the generated text based on a pre-built parser of a context-free grammar. For example, to generate text that follows a simple grammar:

```python
from lark import Lark
import openai
from keymaker.models import gpt4
from keymaker.constraints import ParserConstraint

sql_grammar = """
    start: statement+

    statement: create_table | select_statement

    create_table: "CREATE" "TABLE" ("a" | "b") "(" ("x" | "y") ")"

    select_statement: "SELECT " ("x" | "y") " FROM " ("a" | "b")
"""

parser = Lark(sql_grammar)

constraint = ParserConstraint(parser = parser)
# or pass the grammar directly
# constraint = ParserConstraint(grammar = grammar)

openai.api_key = "..."

model = gpt4()

prompt = Prompt("""
%system%You are a sql expert%/system%
%user%Write me a query that selects the column y from table b.%/user%
""")

prompt = await prompt.complete(model=model, constraint=constraint, name='query', max_tokens=100)
# Prompt('
# %system%You are a sql expert%/system%
# %user%Write me a query that selects the column y from table b.%/user%
# SELECT y FROM b')
```

#### JsonConstraint
```python
from keymaker.constraints import JsonConstraint
```

#### OptionsConstraint

`OptionsConstraint` allows you to constrain the generated text based on a list of string options. For example, to generate text that contains one of the following options:

```python
from keymaker.constraints import OptionsConstraint

options = {"apple", "banana", "orange"}
constraint = OptionsConstraint(options=options)
```

To apply this constraint, pass it to the `complete` method:

```python
prompt = Prompt("I would like an ")
prompt = await prompt.complete(model=hf, constraint=constraint, name="fruit")
print(prompt)
# I would like an apple
```

#### StopsConstraint

`StopsConstraint` allows you to constrain the generated text by stopping at a specified string or regex pattern.

Say we want the model to generate between two XML tags and stop once it reaches the second.

If we are afraid of a malformed end tag with unneeded whitespace, we can account for it as well.

```python
constraint = StopsConstraint(r"<\s*/?\s*hello\s*>", include=True)

prompt = Prompt(
    "Finish this phrase with an end tag then say 'finished' <hello>Hi, the world is "
)

prompt = await prompt.complete(
    model=chat_model, constraint=constraint, name="world_description", stream=MyStream()
)

print(prompt.completions.world_description)
# beautiful.</hello>
```

#### Combining Constraints

Keymaker also allows you to combine multiple constraints using logical operators like `AndConstraint`, `OrConstraint`, and `NotConstraint`.

```python
from keymaker.constraints import OrConstraint, RegexConstraint, OptionsConstraint

regex_constraint = RegexConstraint(pattern=r"peanut")
options_constraint = OptionsConstraint(options={"apple", "banana", "orange"})

combined_constraint = OrConstraint([regex_constraint, options_constraint])

prompt = Prompt("Whenever I see a basketball, it reminds me of my favorite fruit the ")

prompt = (await prompt.complete(model=chat_model, constraint=combined_constraint)) + "."

print(prompt)
# Whenever I see a basketball, it reminds me of my favorite fruit the orange.
```

### Transforming Completions

Sometimes, the output of a completion is not desired to be text from the model.

Simply pass a prompt `complete` an asynchronous function

```python
from keymaker import Completion, CompletionConfig
from keymaker.constraints import RegexConstraint

async def my_stream(completion: Optional[Completion]):
    print(completion)

prompt = await Prompt("10+5={}").format(CompletionConfig(model=..., constraint=RegexConstraint(r"[0-9]", terminate_on_match=False), map_fn=int))

prompt.completions[0].value==15
# True

### Streaming Completions

Keymaker provides a very slim and intuitive means to access completion generation as it happens.

Simply pass a prompt `complete` an asynchronous function

```python
from typing import Optional
from keymaker import Completion

async def my_stream(completion: Optional[Completion]):
    print(completion)

prompt = Prompt("Hello, I'm a talking dog and my name is")

prompt = await prompt.complete(
    model=chat_model,
    stream=my_stream,
)

# R
# over
# .
#  How
#  can
#  I
#  assist
#  you
#  today
# ?
# None
```

As you can see, the incremental tokens `R, over, ...` were passed to the `my_stream` function and were printed as they were generated.
Further, the stream was fed a terminal signal of `None` indicated the stream was complete hence the `Optional[Completion]` type hint.

### Decoding Parameters

Keymaker allows you to set some of the parameters used to sample tokens.

```python
from keymaker.types import Decoder, DecodingStrategy

decoder = Decoder(temperature = 0.7, top_p = 0.95, strategy = DecodingStrategy.GREEDY)
...
# use your parameterization in a completion

prompt = await prompt.complete(..., decoder = decoder)
```


### Creating Custom Models

To create a custom model, you need to extend the `Model` class provided by Keymaker and implement the required methods. Here's an example of creating a custom model:

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

You can then use your custom model with Keymaker as you would with the built-in models:

```python
model = CustomModel(...)
prompt = Prompt("My custom model says: ")
prompt = await prompt.complete(model=model, constraint=your_constraint, name="custom_output")
print(prompt)
```

### Creating Custom Constraints

To create a custom constraint, you need to extend the `Constraint` class provided by Keymaker and implement the `constrain_tokens` method. Here's an example of creating a custom constraint:

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

You can then use your custom constraint with Keymaker as you would with the built-in constraints:

```python
constraint = CustomConstraint(...)
prompt = Prompt("My custom constraint says: ")
prompt = await prompt.complete(model=your_model, constraint=constraint, name="custom_constraint_output")
print(prompt)
```

## Contributing

Contributions are very welcome. Simply fork the repository and open a pull request!

## Acknowledgements

Some constraints in Keymaker are derived from the work of [Matt Rickard](https://github.com/r2d4). Specifically, ReLLM and ParserLLM.
Similar libraries such as [LMQL](https://github.com/eth-sri/lmql) and [Guidance](https://github.com/microsoft/guidance) have surved as motiviation.

## Disclaimer

Keymaker and its contributors bear no responsibility for any harm done by its usage either directly or indirectly including but not limited to costs incurred by using the package (Keymaker) with LLM vendors. The package is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## Copyright
Copyright 2023- Nick Ouellet (nick@ouellet.dev)
