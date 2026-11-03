"""Append-only JSONL storage. The only persistence in the system — no SQLite."""
import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def append(path, record):
    """Append one record as a JSONL line, durably."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def read_all(path):
    """Read every valid JSONL record; skips lines that fail to parse."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return records


def latest_by(records, key_fields):
    """Last write wins: collapse records on a tuple of fields, preserving order."""
    out = {}
    for r in records:
        out[tuple(r.get(k) for k in key_fields)] = r
    return list(out.values())
