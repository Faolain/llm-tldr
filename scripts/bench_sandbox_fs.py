#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_payload() -> dict[str, object]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("expected JSON payload on stdin")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object payload")
    return payload


def _read_window(payload: dict[str, object]) -> int:
    path = Path(str(payload.get("path") or ""))
    if not path.is_absolute():
        raise ValueError("path must be absolute")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(payload.get("start_line") or 1))
    end = min(len(lines), int(payload.get("end_line") or len(lines)))
    body = "\n".join(f"{idx + 1:>5}: {lines[idx]}" for idx in range(start - 1, end))
    sys.stdout.write(body)
    return 0


def _replace_text(payload: dict[str, object]) -> int:
    path = Path(str(payload.get("path") or ""))
    if not path.is_absolute():
        raise ValueError("path must be absolute")
    old = str(payload.get("old") or "")
    new = str(payload.get("new") or "")
    count = int(payload.get("count") or 1)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError("old text not found")
    replaced = text.replace(old, new, count if count >= 0 else text.count(old))
    path.write_text(replaced, encoding="utf-8")
    sys.stdout.write("replace_text applied")
    return 0


def _write_file(payload: dict[str, object]) -> int:
    path = Path(str(payload.get("path") or ""))
    if not path.is_absolute():
        raise ValueError("path must be absolute")
    content = str(payload.get("content") or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    sys.stdout.write("write_file applied")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sandbox-local file helper for benchmark containers.")
    ap.add_argument("operation", choices=["read-window", "replace-text", "write-file"])
    args = ap.parse_args()

    payload = _load_payload()
    if args.operation == "read-window":
        return _read_window(payload)
    if args.operation == "replace-text":
        return _replace_text(payload)
    if args.operation == "write-file":
        return _write_file(payload)
    raise ValueError(f"unsupported operation: {args.operation}")


if __name__ == "__main__":
    raise SystemExit(main())
