from typing import TYPE_CHECKING, Dict, FrozenSet, List

import regex as re

if TYPE_CHECKING:
    from keymaker import Prompt


def split_tags(
    text: str,
    tag_start: str = "%",
    tag_end: str = "%",
    default_role: str = "assistant",
    roles: FrozenSet[str] = frozenset(("system", "user", "assistant")),
) -> List[Dict[str, str]]:
    """
    Splits a text string into a list of messages based on tags.

    Args:
        text (str): The input text to split into messages.
        tag_start (str, optional): The start delimiter for tags. Defaults to '%'.
        tag_end (str, optional): The end delimiter for tags. Defaults to '%'.
        default_role (str, optional): The default role to use for untagged messages. Defaults to 'assistant'.
        roles (FrozenSet[str], optional): The list of valid roles for tagged messages. Defaults to ['system', 'user', 'assistant'].

    Returns:
        List[Dict[str, str]]: A list of messages, where each message is a dictionary with keys 'role' and 'content'.

    Raises:
        Exception: If an end tag is found with no start tag, or if an unknown or mismatched tag is found.

    Examples:
        >>> split_tags('\nYou are a friendly bot\n%/system%\n%user%Can you help me calculate stuff?%/user%\nYes, how may I help you?\n%user%\nI want to know the square root of 10%/user%\n%assistant%\nSure the square root of 10 is ...\n%/assistant%')
        [{'role': 'assistant', 'content': 'You are a friendly bot\n'}, {'role': 'system', 'content': ''}, {'role': 'user', 'content': 'Can you help me calculate stuff?'}, {'role': 'assistant', 'content': 'Yes, how may I help you?\n'}, {'role': 'user', 'co...
    """
    text = str(text)
    messages = []

    while text:
        # first we check to see if the text is untagged
        match_role = None
        match = re.search(
            rf"(?P<content>\s*.*?)\s*(?P<tag>{tag_start}/?(?P<role>.*?){tag_end}\s*|$)",
            text,
        )
        if match:
            content = match.group("content")
            if match.group("tag").startswith(f"{tag_start}/"):
                raise Exception(f"Found end tag with no start `{match.group('tag')}`.")
            if match.group("role") is not None and match.group("role") not in roles:
                raise Exception(f"Unknown role `{match.group('role')}`.")
            if content.strip():
                messages.append({"role": default_role, "content": content.strip()})
            text = text[match.span("tag")[1] :]  # noqa: E203
            match_role = match.group("role")
        if not text:
            break
        # now that we have defaulted any untagged text, we can handle the next tagged portion
        match = re.search(rf"\s*.*?(?P<tag>{tag_start}/?(?P<role>.*?){tag_end}\s*)", text)
        content = text[: match.span("tag")[0]]
        if (match_role is not None and match.group("role") != match_role) or (not match.group("tag").startswith(f"{tag_start}/")):
            raise Exception(f"Unclosed tag `{match_role}`. Found `{match.group('role')}`.")
        messages.append({"role": match_role, "content": content.strip()})
        text = text[match.span("tag")[1] :]  # noqa: E203

    return messages


def strip_tags(
    prompt: "Prompt",
    tag_start: str = "%",
    tag_end: str = "%",
    roles_seps: Dict[str, str] = {
        "system": "",
        "user": "User: ",
        "assistant": "Assistant: ",
    },
    sep: str = "\n",
) -> "Prompt":
    from keymaker import Prompt

    messages = split_tags(text=prompt, tag_start=tag_start, tag_end=tag_end, roles=frozenset(roles_seps.keys()))
    return Prompt(
        sep.join((roles_seps[message["role"]] + message["content"] for message in messages)),
        prompt.completions,  # type: ignore
    )
