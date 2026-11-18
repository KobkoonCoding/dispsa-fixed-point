"""Section 4.1: affine split feasibility problem; Table 3 and the error values quoted for (kappa_1, kappa_2) = (8, 3)."""
from __future__ import annotations

import sys

import numpy as np

from .algorithms import EVALS_BOT, EVALS_DISPSA, evals_to_reach, run_bot, run_dispsa
from .common import bot_key, save_json, save_table
from .problems import make_affine_sfp
from .settings import BOT_GRID, SFP_BETA, SFP_MAX_EVALS, SFP_SEED, SFP_SETTING, SFP_TOL

CURVE_EVALS = 600


def counts(p) -> dict:
    out = {}
    for k, t, x0 in p.starts:
        hit = lambda x: p.rel_error(x) <= SFP_TOL
        row = {"DISPSA": evals_to_reach(lambda N, cb: run_dispsa(p.T, x0, N, SFP_BETA, SFP_SETTING, cb),
                                        EVALS_DISPSA, SFP_MAX_EVALS, hit)}
        for lam in BOT_GRID:
            row[bot_key(lam)] = evals_to_reach(lambda N, cb, lam=lam: run_bot(p.T, x0, N, SFP_BETA, lam, cb),
                                               EVALS_BOT, SFP_MAX_EVALS, hit)
        # Without Tikhonov regularization (beta_n = 1, lambda_n = 1): x_{n+1} = T x_n.
        x, _ = run_bot(p.T, x0, SFP_MAX_EVALS, lambda n: 1.0, 1.0)
        row["beta=1 error"] = p.rel_error(x)
        P = p.M.T @ p.M
        row["null-space part of x0"] = float(np.linalg.norm(x0 - P @ x0) / max(1.0, np.linalg.norm(p.x_dagger)))
        out[f"({k},{t})"] = row
    return out


def curves(p, case=(8, 3)) -> dict:
    x0 = next(x for k, t, x in p.starts if (k, t) == case)
    res = {"DISPSA": [(0, p.rel_error(x0))]}
    run_dispsa(p.T, x0, CURVE_EVALS // EVALS_DISPSA, SFP_BETA, SFP_SETTING,
               lambda n, x: res["DISPSA"].append((EVALS_DISPSA * n, p.rel_error(x))) and False)
    for lam in (1.0, 1.4):
        key = bot_key(lam)
        res[key] = [(0, p.rel_error(x0))]
        run_bot(p.T, x0, CURVE_EVALS, SFP_BETA, lam, lambda n, x, key=key: res[key].append((n, p.rel_error(x))) and False)
    res["beta=1"] = [(0, p.rel_error(x0))]
    run_bot(p.T, x0, CURVE_EVALS, lambda n: 1.0, 1.0, lambda n, x: res["beta=1"].append((n, p.rel_error(x))) and False)
    return res


def main(root: str = "results") -> None:
    p = make_affine_sfp(SFP_SEED)
    c = counts(p)
    save_json(root, "sfp_results.json", {"counts": c, "dim_solution_set": 240 - 42,
                                         "norm_x_dagger": float(np.linalg.norm(p.x_dagger))})
    rows = [f"$({k},{t})$ & {c[f'({k},{t})']['DISPSA']} & {c[f'({k},{t})']['Bot (1)']} & "
            f"{c[f'({k},{t})']['Bot (1.4)']} & {c[f'({k},{t})']['beta=1 error']:.3f} \\\\"
            for k, t, _ in p.starts]
    save_table(root, "table3_sfp.tex", rows)
    res = curves(p)
    save_json(root, "sfp_curves.json", res)
    print({key: [r[key] for r in c.values()] for key in ["DISPSA"] + [bot_key(l) for l in BOT_GRID]})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
