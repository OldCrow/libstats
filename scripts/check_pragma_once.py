"""Check that every header named on the command line uses #pragma once.

This is the libstats convention for header guards. Exit 1 if any file lacks
a line starting with "#pragma once"; skips arguments that are not files.
"""

import sys
from pathlib import Path


def has_pragma_once(path: Path) -> bool:
    # Bytes, split on "\n" only: the same lines grep sees.
    return any(line.startswith(b"#pragma once") for line in path.read_bytes().split(b"\n"))


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    exit_code = 0
    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue  # might be deleted
        if not has_pragma_once(path):
            print(f"ERROR: {name} is missing '#pragma once'")
            print("       libstats uses #pragma once as the header guard convention")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
