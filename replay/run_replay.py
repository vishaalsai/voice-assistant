"""
CLI entry point for the replay system.

Usage:
    python replay/run_replay.py list            — list all saved replays
    python replay/run_replay.py {run_id}        — re-run a saved replay
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from replay.replay import list_replays, run_replay

USAGE = """\
Voice Assistant — Replay CLI

Usage:
  python replay/run_replay.py list         List all saved replay runs
  python replay/run_replay.py <run_id>     Re-execute a specific run

Examples:
  python replay/run_replay.py list
  python replay/run_replay.py a3f9c1
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    command = sys.argv[1]

    if command == "list":
        list_replays()
    else:
        asyncio.run(run_replay(command))
