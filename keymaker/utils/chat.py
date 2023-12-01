from typing import TYPE_CHECKING, Dict, FrozenSet, List

if TYPE_CHECKING:
    from keymaker import Prompt

from parsy import ParseError, any_char, eof, generate, peek, string, string_from


def split_tags(
    text: str,
    tag_start: str = "%",
    tag_end: str = "%",
    default_role: str = "assistant",
    roles: FrozenSet[str] = frozenset(("system", "user", "assistant")),
    discard_default_whitespace: bool = True,
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

    tag_begin = string(tag_start) >> string_from(*roles) << string(tag_end)

    @generate
    def tag_parser():
        role = yield tag_begin
        tag_complete = (string(tag_start + "/") >> string(role) << string(tag_end)) | peek(eof)
        content = yield any_char.until(tag_complete)
        yield tag_complete
        return {"role": role, "content": "".join(content)}

    default = any_char.until(tag_begin | eof)

    @generate
    def message_parser():
        tot = []
        while True:
            rec = None
            try:
                rec = yield default
                if rec:
                    content = "".join(rec)
                    if not discard_default_whitespace or content.strip():
                        tot.append({"role": default_role, "content": content})
                    continue
            except ParseError:
                pass
            try:
                rec = yield tag_parser.many()
                if rec:
                    tot += rec
                    continue
            except ParseError:
                pass
            break

        return tot

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
