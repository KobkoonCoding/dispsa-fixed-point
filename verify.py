"""Compare a new run with the shipped outputs.  Usage: python verify.py results expected_results

Table files must be identical; every number in the JSON files must agree to a relative tolerance of 1e-6."""
import json
import math
import sys
from pathlib import Path


def walk(a, b, path, bad):
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: missing")
            else:
                walk(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12):
            bad.append(f"{path}: {a} != {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} != {b!r}")


def main(new: str, ref: str) -> int:
    new_p, ref_p = Path(new), Path(ref)
    bad = []
    for t in sorted((ref_p / "tables").glob("*.tex")):
        other = new_p / "tables" / t.name
        if not other.exists() or other.read_text(encoding="utf-8") != t.read_text(encoding="utf-8"):
            bad.append(f"tables/{t.name} differs")
    for j in sorted(ref_p.glob("*.json")):
        other = new_p / j.name
        if not other.exists():
            bad.append(f"{j.name} missing")
            continue
        walk(json.loads(other.read_text(encoding="utf-8")), json.loads(j.read_text(encoding="utf-8")), j.name, bad)
    print("\n".join(bad[:50]) if bad else "All tables and numbers agree with the expected results.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results",
                  sys.argv[2] if len(sys.argv) > 2 else "expected_results"))
