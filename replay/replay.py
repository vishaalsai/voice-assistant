"""
Replay system for deterministic pipeline debugging.
Saves input audio and metadata so any past run can be reproduced exactly,
and provides utilities to list and re-execute saved replays.
"""

import asyncio
import json
import shutil
import time
from pathlib import Path

# Canonical location for all saved replay files.
REPLAY_DIR = Path(__file__).parent / "recorded_inputs"


def save_replay(audio_path: str, run_id: str) -> None:
    """
    Copy the input audio file and write a metadata sidecar into REPLAY_DIR.

    Args:
        audio_path: Path to the original .wav file used for this run.
        run_id:     Short unique identifier for this run (e.g. uuid hex[:6]).
    """
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    dest_wav = REPLAY_DIR / f"{run_id}.wav"
    shutil.copy2(audio_path, dest_wav)

    meta = {
        "run_id":              run_id,
        "original_audio_path": str(Path(audio_path).resolve()),
        "saved_at":            time.time(),
        "purpose":             "replay input for deterministic debugging",
    }
    dest_json = REPLAY_DIR / f"{run_id}.json"
    dest_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


async def run_replay(run_id: str) -> None:
    """
    Re-run the full pipeline using the audio saved under run_id.

    Looks up the .wav file in REPLAY_DIR; prints a clear error if it does
    not exist, otherwise delegates to run_pipeline() in main.py.

    Args:
        run_id: The identifier used when save_replay() was originally called.
    """
    # Import here to avoid a circular import at module load time
    # (main imports replay, replay imports main).
    from main import run_pipeline

    wav_path = REPLAY_DIR / f"{run_id}.wav"

    if not wav_path.exists():
        print(f"[Replay] No replay found for run_id: {run_id}")
        print(f"         Expected file: {wav_path}")
        return

    print(f"🔁 Replaying run: {run_id}")
    await run_pipeline(str(wav_path))


def list_replays() -> None:
    """
    Print a formatted table of all saved replays in REPLAY_DIR.
    Reads each .json sidecar for metadata; skips entries with missing files.
    """
    json_files = sorted(REPLAY_DIR.glob("*.json"))

    if not json_files:
        print("[Replay] No saved replays found.")
        return

    col_id       = 10
    col_saved    = 26
    col_path     = 44

    header  = f"{'run_id':<{col_id}}  {'saved_at':<{col_saved}}  {'original_audio_path':<{col_path}}"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    for jf in json_files:
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        run_id    = meta.get("run_id", jf.stem)
        saved_ts  = meta.get("saved_at", 0)
        saved_fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_ts))
        orig_path = meta.get("original_audio_path", "unknown")

        # Truncate long paths so the table stays readable.
        if len(orig_path) > col_path:
            orig_path = "..." + orig_path[-(col_path - 3):]

        print(f"{run_id:<{col_id}}  {saved_fmt:<{col_saved}}  {orig_path:<{col_path}}")

    print(divider)
