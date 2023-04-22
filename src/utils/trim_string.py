import re


def trim_string(string: str, fill: str = ' ') -> str:
    return fill.join(filter(lambda _: _, re.split(r"\W", string))).lower()