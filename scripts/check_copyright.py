"""Warn about source files with no copyright notice in their first 10 lines.

Warn-only: always exits 0, as the shell script it replaces did.
"""

import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path

# Matched against the whole path as given; "*" crosses "/". Names without a
# leading "*" (.gitignore, LICENSE, ...) therefore match only at the repo root.
SKIP_PATTERNS = (
    "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
    ".gitignore", ".clang*", ".cmake*", "LICENSE", "README",
)  # fmt: skip
NOTICE = re.compile(r"(Copyright|copyright|©).*[0-9]{4}")


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:  # stands in for `file | grep binary`
            continue
        if any(fnmatchcase(name, pattern) for pattern in SKIP_PATTERNS):
            continue
        head = data.split(b"\n")[:10]
        if not any(NOTICE.search(line.decode("utf-8", "replace")) for line in head):
            print(f"WARNING: {name} may be missing copyright header")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
