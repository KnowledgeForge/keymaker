---
categories: ["docs"]
weight: 50
title: Using Constraints
tags: ["how-to", "core"]
series: ["Usage"]
series_weight: 10
lightgallery: true
---
# Using Constraints

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