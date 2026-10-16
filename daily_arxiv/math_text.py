"""Shared formula boundaries for translation, validation, and HTML rendering."""
import re

# Inline math may have inner whitespace, touch digits, or be adjacent to another
# formula ($3.88$$\\times$). Only prose prices are excluded at the opening
# delimiter; restrictions on closing dollars can silently swallow whole sentences.
MATH_PATTERN = (r"(?<!\\)\$\$.*?(?<!\\)\$\$|"
                r"(?<!\\)\$(?!\$)(?!\d[\d,.]*(?:[KMBkmb])?\s+[A-Za-z])"
                r"(?:\\+\$|\\.|[^\\$])*?(?<!\\)\$|"
                r"\\\(.*?\\\)|\\\[.*?\\\]")

def normalize_math_source(text):
    # Some abstracts escape the closing dollar of a standalone command, e.g.
    # "$ \\gamma\\$ limit". Repair that unambiguous typo for translation and display;
    # otherwise the opening dollar captures the following English paragraph.
    return re.sub(r'(?<!\\)\$(\s*\\[A-Za-z]+)\\\$(?=\s)', r'$\1$', text)
