"""Warn about include-order violations in .cpp/.cc/.cxx files.

The order checked is the one .clang-format enforces (IncludeBlocks: Regroup
with its IncludeCategories), so this hook and clang-format cannot disagree:

  1. the file's own header (the first quoted include)
  2. project headers: every other quoted include
  3. angle-bracket headers, and quoted gtest/gmock/isl/json headers

clang-format sorts each run of includes on its own, and only blank lines
join two runs: the order restarts after any other line (a comment, an #if,
a macro, code).

Warn-only: always exits 0, as the shell script it replaced did.

History: that script expected own, C, C++, libraries, project. It matched C
headers as <c[a-z]+>, which also caught <chrono>, <concepts> and <charconv>,
and its order was the reverse of the one clang-format applies here, so it
warned on almost every correctly formatted file.
"""

import re
import sys
from pathlib import Path

INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
THIRD_PARTY_QUOTED = re.compile(r"^(gtest|gmock|isl|json)/")
CATEGORY_NAMES = {1: "own header", 2: "project header", 3: "system/library header"}


def categorize(opener: str, target: str, first_in_file: bool) -> int:
    if opener == "<" or THIRD_PARTY_QUOTED.match(target):
        return 3
    return 1 if first_in_file else 2


def check_include_order(name: str, text: str) -> int:
    warnings = 0
    last_category = 0
    seen_include = False
    for line_num, line in enumerate(text.split("\n"), 1):
        match = INCLUDE.match(line)
        if not match:
            if line.strip():
                last_category = 0  # a new run of includes starts after this
            continue
        category = categorize(match.group(1), match.group(2), not seen_include)
        seen_include = True
        if category < last_category:
            warnings += 1
            print(f"WARNING: {name}:{line_num} - Include order violation")
            print(
                f"  {CATEGORY_NAMES[category]} after {CATEGORY_NAMES[last_category]}; "
                "expected own header, project headers, then <...> headers"
            )
            print(f"  Line: {line.rstrip()}")
        last_category = category
    return warnings


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue
        if name.endswith((".cpp", ".cc", ".cxx")):
            check_include_order(name, path.read_bytes().decode("utf-8", "replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
