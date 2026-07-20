import re

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_headings(text):
    out = []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            out.append({"level": len(m.group(1)), "title": m.group(2).strip()})
    return out
