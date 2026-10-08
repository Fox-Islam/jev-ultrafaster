"""Jev chooses an observed action. Code owns execution."""

import os
import secrets

# One daemon serves one run at a time, so runs side by side each need their own. The harness reads
# BU_NAME when it is first imported, which is why this is decided here, before anything imports it.
# Each daemon is a process of its own and is stopped when the process that started it leaves.
if os.environ.get("JEV_DAEMON_PER_RUN") == "1":
    os.environ["BU_NAME"] = f"jev-{os.getpid()}-{secrets.token_hex(3)}"

from .agent import Agent  # noqa: E402
from .browser import Browser  # noqa: E402

__all__ = ["Agent", "Browser"]
