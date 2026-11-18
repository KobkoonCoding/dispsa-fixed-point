"""Output helpers shared by the experiment modules."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.titlesize": 9,
                            "legend.fontsize": 8})


def outdir(root: str | Path, *sub: str) -> Path:
    p = Path(root, *sub)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(root: str | Path, name: str, obj) -> None:
    with open(outdir(root) / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def save_table(root: str | Path, name: str, rows: list[str]) -> None:
    with open(outdir(root, "tables") / name, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")


def bot_key(lam: float) -> str:
    return f"Bot ({lam:g})"
