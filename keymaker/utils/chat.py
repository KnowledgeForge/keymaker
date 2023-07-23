from typing import TYPE_CHECKING, Dict, FrozenSet, List

if TYPE_CHECKING:
    from keymaker import Prompt

from parsy import alt, regex, seq, string


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
        >>> split_tags('%system%\nYou are a friendly bot\n%/system%\n%user%Can you help me calculate stuff?%/user%\nYes, how may I help you?\n%user%\nI want to know the square root of 10%/user%\n%assistant%\nSure the square root of 10 is ...\n%/assistant%')
        [{'role': 'assistant', 'content': 'You are a friendly bot\n'}, {'role': 'system', 'content': ''}, {'role': 'user', 'content': 'Can you help me calculate stuff?'}, {'role': 'assistant', 'content': 'Yes, how may I help you?\n'}, {'role': 'user', 'co...
    """
    text = str(text)
    role = alt(*(string(s) for s in roles))
    tag_begin = seq(
        string(tag_start),
        role,
        string(tag_end),
    )
    tag_complete = seq(
        string(tag_start),
        string("/"),
        role,
        string(tag_end),
    )

    content = regex(r"([^%]+|\n(?!\s*%))+")

    message = seq(tag_begin, content, tag_complete).map(lambda x: {'role': x[0][1], 'content': x[1]}) | content.map(
        lambda x: {'role': default_role, 'content': x},
    )

    message_parser = message.at_least(1)
    return message_parser.parse(text)


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
