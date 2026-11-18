"""Section 4.3: breast cancer classification by sparse logistic regression; Table 6 and the first row of Table 7."""
from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .algorithms import EVALS_BOT, EVALS_DISPSA, evals_to_reach, run_bot, run_dispsa
from .common import bot_key, save_json, save_table
from .problems import make_logistic
from .settings import BOT_GRID, CLF_BUDGET, CLF_MAX_EVALS, CLF_RES_FACTOR, CLF_SEEDS, COMP_BETA, COMP_SETTING

METRICS = ("accuracy", "precision", "recall", "f1", "auc")


def metrics(p, w) -> dict:
    score = p.A_test @ w
    pred = (score > 0).astype(int)
    return {"accuracy": 100.0 * accuracy_score(p.y_test, pred),
            "precision": precision_score(p.y_test, pred, zero_division=0),
            "recall": recall_score(p.y_test, pred, zero_division=0),
            "f1": f1_score(p.y_test, pred, zero_division=0),
            "auc": roc_auc_score(p.y_test, score)}


def one_split(seed: int) -> dict:
    p = make_logistic(seed)
    T = p.T_step(0.995 / p.L)
    w0 = np.zeros(p.A.shape[1])
    r0 = float(np.linalg.norm(w0 - T(w0)))
    hit = lambda w: np.linalg.norm(w - T(w)) <= CLF_RES_FACTOR * r0
    row = {"seed": seed, "m_train": int(p.A.shape[0]), "m_test": int(p.A_test.shape[0]), "L": p.L,
           "evals DISPSA": evals_to_reach(lambda N, cb: run_dispsa(T, w0, N, COMP_BETA, COMP_SETTING, cb),
                                          EVALS_DISPSA, CLF_MAX_EVALS, hit)}
    for lam in BOT_GRID:
        row[f"evals {bot_key(lam)}"] = evals_to_reach(lambda N, cb, lam=lam: run_bot(T, w0, N, COMP_BETA, lam, cb),
                                                      EVALS_BOT, CLF_MAX_EVALS, hit)
    wd, _ = run_dispsa(T, w0, CLF_BUDGET // EVALS_DISPSA, COMP_BETA, COMP_SETTING)
    row["DISPSA"] = {**metrics(p, wd), "objective": p.objective(wd)}
    for lam in BOT_GRID:
        wb, _ = run_bot(T, w0, CLF_BUDGET, COMP_BETA, lam)
        row[bot_key(lam)] = {**metrics(p, wb), "objective": p.objective(wb)}
    return row


def main(root: str = "results") -> None:
    rows = [one_split(s) for s in CLF_SEEDS]
    summary = {}
    for key in ["DISPSA"] + [bot_key(l) for l in BOT_GRID]:
        summary[key] = {m: (float(np.mean([r[key][m] for r in rows])), float(np.std([r[key][m] for r in rows], ddof=1)))
                        for m in METRICS + ("objective",)}
        v = [r[f"evals {key}"] for r in rows]
        summary[key]["evals"] = (float(np.mean(v)), int(min(v)), int(max(v)), int(sum(v)))
    t4 = []
    for key, name in (("DISPSA", "DISPSA"), ("Bot (1)", r"Bo\c{t} (1)"), ("Bot (1.4)", r"Bo\c{t} (1.4)")):
        s = summary[key]
        t4.append(f"{name} & {s['accuracy'][0]:.2f} $\\pm$ {s['accuracy'][1]:.2f} & "
                  + " & ".join(f"{s[m][0]:.4f} $\\pm$ {s[m][1]:.4f}" for m in METRICS[1:]) + " \\\\")
    t5 = [" & ".join([f"{summary[k]['evals'][0]:.1f} ({summary[k]['evals'][1]}--{summary[k]['evals'][2]})"
                      for k in ("DISPSA", "Bot (1)", "Bot (1.4)")]) + " \\\\"]
    save_table(root, "table6_classification.tex", t4)
    save_table(root, "table7_row1_constant_step.tex", t5)
    save_json(root, "classification_results.json", {"summary": summary, "runs": rows})
    print({k: summary[k]["evals"] for k in summary})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
