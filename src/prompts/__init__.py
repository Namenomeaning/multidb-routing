"""Prompt registry: each prompt is a .txt file loaded verbatim.

Keeping prompts as standalone text (rather than inline Python strings) makes them
easy to inspect, diff, and cite in the paper -- mirroring the Sudarshan
DB-Routing-prompts layout. Load with `load("<name>")`.
"""
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(name: str) -> str:
    """Return the raw text of prompts/<name>.txt (byte-for-byte)."""
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8")
