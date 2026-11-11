"""Warn about debug code that should not be committed.

Warn-only: always exits 0, as the shell script it replaces did.
"""

import re
import sys
from pathlib import Path

DEBUG_PATTERNS = [
    r"std::cout.*<<.*DEBUG",
    r"printf.*DEBUG",
    r"console\.log",
    r"#define\s+DEBUG\s+1",
    r"// *TODO:.*remove",
    r"// *FIXME:.*temporary",
    r"// *HACK:",
    r"// *XXX:",
    r"DO NOT COMMIT",
    r"DO NOT SUBMIT",
]
COMPILED = [re.compile(p) for p in DEBUG_PATTERNS]
# std::cout is allowed in these; matched anywhere in the path, as before.
COUT_EXEMPT = re.compile(r"(test|example|tool|benchmark)")


def numbered_matches(lines: list[str], pattern: re.Pattern[str]) -> list[str]:
    return [f"{i}:{line}" for i, line in enumerate(lines, 1) if pattern.search(line)]


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue
        lines = path.read_bytes().decode("utf-8", "replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # text after the last newline only if non-empty, like grep
        for pattern in COMPILED:
            matches = numbered_matches(lines, pattern)
            if matches:
                print(f"WARNING: {name} contains potential debug code:")
                print("\n".join(matches))
                print()
        if not COUT_EXEMPT.search(name):
            couts = numbered_matches(lines, re.compile(r"std::cout"))
            if couts:
                print(f"WARNING: {name} contains std::cout (should use proper logging)")
                print("  Found at: " + "\n".join(couts[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
