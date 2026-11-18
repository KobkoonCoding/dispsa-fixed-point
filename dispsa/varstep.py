"""Section 4.3: variable step sizes r_n = (1 + 0.2/(n+1))/L in the classification problem; second row of Table 7."""
from __future__ import annotations

import sys

import numpy as np

from .algorithms import EVALS_BOT, EVALS_DISPSA, evals_to_reach, run_bot, run_dispsa
from .common import bot_key, save_json, save_table
from .problems import make_logistic
from .settings import CLF_MAX_EVALS, CLF_RES_FACTOR, CLF_SEEDS, COMP_BETA, COMP_SETTING

LAMS = (1.0, 1.4)


def step(n: int, L: float) -> float:
    return (1.0 + 0.2 / (n + 1)) / L


def one_split(seed: int) -> dict:
    p = make_logistic(seed)
    T = p.T_step(0.995 / p.L)               # constant step size r = 0.995/L, used only in the stopping rule
    Tn = lambda n: p.T_step(step(n, p.L))
    w0 = np.zeros(p.A.shape[1])
    r0 = float(np.linalg.norm(w0 - T(w0)))
    hit = lambda w: np.linalg.norm(w - T(w)) <= CLF_RES_FACTOR * r0
    row = {"seed": seed, "DISPSA": evals_to_reach(
        lambda N, cb: run_dispsa(T, w0, N, COMP_BETA, COMP_SETTING, cb, Tn=Tn), EVALS_DISPSA, CLF_MAX_EVALS, hit)}
    for lam in LAMS:
        row[bot_key(lam)] = evals_to_reach(lambda N, cb, lam=lam: run_bot(T, w0, N, COMP_BETA, lam, cb, Tn=Tn),
                                           EVALS_BOT, CLF_MAX_EVALS, hit)
    return row


def main(root: str = "results") -> None:
    rows = [one_split(s) for s in CLF_SEEDS]
    keys = ["DISPSA"] + [bot_key(l) for l in LAMS]
    summary = {k: (float(np.mean([r[k] for r in rows])), int(min(r[k] for r in rows)), int(max(r[k] for r in rows)))
               for k in keys}
    fewer = {k: int(sum(r["DISPSA"] < r[k] for r in rows)) for k in keys[1:]}
    save_table(root, "table7_row2_variable_step.tex",
               [" & ".join(f"{summary[k][0]:.1f} ({summary[k][1]}--{summary[k][2]})" for k in keys) + " \\\\"])
    save_json(root, "varstep_results.json", {"summary": summary, "dispsa_fewer_in_runs": fewer, "runs": rows,
                                             "max_step_times_L": step(1, 1.0), "sum_abs_step_diff_times_L":
                                             step(1, 1.0) - 1.0})
    print(summary, fewer)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
