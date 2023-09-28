"""
Fixtures for testing.
"""
from typing import Iterator

import pytest

from keymaker.models import Huggingface


@pytest.fixture
def distilgpt2() -> Iterator[Huggingface]:
    """
    simple model for testing
    """

    yield Huggingface("distilgpt2")
