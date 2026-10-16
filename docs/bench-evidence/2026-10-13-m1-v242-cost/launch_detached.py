#!/usr/bin/env python3
"""Detach the overnight script into its own session so it survives the Claude session ending."""
import os, subprocess, sys
O = "/Users/wolfman/Development/libstats-m-20261013"
out = open(f"{O}/overnight.stdout", "ab")
p = subprocess.Popen(["/bin/bash", f"{O}/overnight_m_20261013.sh"], stdout=out, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=O)
print(f"detached pid={p.pid} sid={os.getsid(p.pid)}")
