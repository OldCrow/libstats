"""Fail on debug code that should not be committed.

Two rules, both enforced (exit 1 on any finding):

1. Leftover markers anywhere in a C/C++ file: DEBUG prints, `#define DEBUG 1`,
   "TODO: ... remove", "FIXME: ... temporary", "HACK:", "XXX:",
   "DO NOT COMMIT", "DO NOT SUBMIT".
2. `std::cout` in library code. A library must not write to stdout. Not
   applied to tests, examples, tools and benchmarks (matched anywhere in the
   path), to comment lines (Doxygen examples), or to code inside a
   preprocessor conditional whose condition names DEBUG, which is how
   deliberate diagnostics are gated (e.g. `#ifdef LIBSTATS_DEBUG_THREADING`).

History: the shell script this replaced printed warnings and always exited 0,
so it could not stop anything, and its std::cout rule flagged documentation.
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
COUT = re.compile(r"std::cout")
COUT_EXEMPT_PATH = re.compile(r"(test|example|tool|benchmark)")
COMMENT_LINE = re.compile(r"^\s*(//|/\*|\*)")
PP_OPEN = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b(.*)")
PP_CLOSE = re.compile(r"^\s*#\s*endif\b")


def check_file(name: str, lines: list[str]) -> int:
    findings = 0
    for pattern in COMPILED:
        for line_num, line in enumerate(lines, 1):
            if pattern.search(line):
                findings += 1
                print(f"{name}:{line_num}: debug marker: {line.strip()}")

    if COUT_EXEMPT_PATH.search(name):
        return findings
    conditions: list[str] = []  # enclosing #if/#ifdef/#ifndef conditions
    for line_num, line in enumerate(lines, 1):
        opened = PP_OPEN.match(line)
        if opened:
            conditions.append(opened.group(2))
            continue
        if PP_CLOSE.match(line):
            if conditions:
                conditions.pop()
            continue
        if not COUT.search(line) or COMMENT_LINE.match(line):
            continue
        if any("DEBUG" in condition for condition in conditions):
            continue
        findings += 1
        print(f"{name}:{line_num}: std::cout in library code: {line.strip()}")
    return findings


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    findings = 0
    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", "replace")
        findings += check_file(name, text.split("\n"))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
