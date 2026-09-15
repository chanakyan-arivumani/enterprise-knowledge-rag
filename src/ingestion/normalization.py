import re


def normalize_text(text: str) -> str:
    """1. Normalize line endings
    2. Remove BOM
    3. Remove trailing whitespace from each line
    4. Remove unnecessary leading/trailing whitespace from the entire document
    """
    text = text.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)
