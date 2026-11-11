"""Warn about include-order violations in .cpp/.cc/.cxx files.

Expected order: own header, C headers, C++ standard headers, other library
headers, project headers. Warn-only: always exits 0, as the shell script it
replaces did. The category regexes keep the shell script's exact (loose)
behaviour, e.g. <chrono> and <concepts> count as C headers (c[a-z]+).
"""

import re
import sys
from pathlib import Path

OWN_HEADER = re.compile(r'#include "[^\n]*\.h"')
C_CXX_STYLE = re.compile(r"#include <c[a-z]+>")
C_SYSTEM = re.compile(r"#include <[a-z_]+\.h>")
CXX_STD = re.compile(r"#include <[a-z_]+>")
OTHER_LIB = re.compile(r"#include <.*>")


def categorize(include_line: str, last_category: int) -> int:
    """1=own header, 2=C, 3=C++, 4=other library, 5=project."""
    if OWN_HEADER.search(include_line) and last_category == 0:
        return 1
    if C_CXX_STYLE.search(include_line) or C_SYSTEM.search(include_line):
        return 2
    if CXX_STD.search(include_line):
        return 3
    if OTHER_LIB.search(include_line):
        return 4
    return 5


def check_include_order(name: str, text: str) -> None:
    last_category = 0
    for line_num, line in enumerate(text.split("\n"), 1):
        if not line.startswith("#include"):
            continue
        category = categorize(line, last_category)
        if category < last_category and category != 1:
            print(f"WARNING: {name}:{line_num} - Include order violation")
            print("  Expected category order: own-header, C, C++, libraries, project")
            print(f"  Line: {line}")
        if category != 1 or last_category == 0:
            last_category = category


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
