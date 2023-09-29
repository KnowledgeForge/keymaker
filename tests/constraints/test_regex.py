"""Testing the options constraint"""

import pytest
import regex as re

from keymaker import CompletionConfig, Prompt
from keymaker.constraints import RegexConstraint


@pytest.mark.asyncio
async def test_options_constraint(distilgpt2):
    prompt = Prompt("An item you would take to the beach: {beach_item}", distilgpt2)
    beach_items_set = re.compile(r"Cooler|Beach Chair|Sunscreen|Volleyball|Suntan Lotion|Drinks")
    beach_items = RegexConstraint(beach_items_set)
    filled = await prompt.format(beach_item=CompletionConfig(constraint=beach_items))
    assert filled.completions["beach_item"] in {"Cooler", "Beach Chair", "Sunscreen", "Volleyball", "Suntan Lotion", "Drinks"}
