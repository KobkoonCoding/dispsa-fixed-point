"""Section 4.4: ablation study of the two inertial terms; Table 8."""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from .algorithms import EVALS_DISPSA, evals_to_reach, run_dispsa
from .common import save_json, save_table
from .deblurring import MAX_EVALS as DEBLUR_MAX, reference
from .problems import haar2, make_affine_sfp, make_deblur, make_logistic
from .settings import (CLF_MAX_EVALS, CLF_RES_FACTOR, CLF_SEEDS, COMP_BETA, COMP_SETTING, DEBLUR_GAP_TOL,
                       DEBLUR_NOISE, DEBLUR_SEEDS, SFP_BETA, SFP_MAX_EVALS, SFP_SEED, SFP_SETTING, SFP_TOL)

VERSIONS = (("No inertia", 0.0, 0.0), ("First inertial term only", None, 0.0),
            ("Second inertial term only", 0.0, None), ("Both inertial terms", None, None))
ROW_LABELS = {"No inertia": r"No inertia, $\mu_n=\rho_n=0$",
              "First inertial term only": r"First inertial term only, $\rho_n=0$",
              "Second inertial term only": r"Second inertial term only, $\mu_n=0$",
              "Both inertial terms": r"Both inertial terms"}


def variant(setting, mu, rho):
    return replace(setting, mu=setting.mu if mu is None else mu, rho=setting.rho if rho is None else rho)


def main(root: str = "results") -> None:
    sfp = make_affine_sfp(SFP_SEED)
    deb = []
    for sigma in DEBLUR_NOISE:
        for seed in DEBLUR_SEEDS:
            p = make_deblur(sigma, seed)
            x0 = haar2(p.observed, p.levels).ravel()
            u = reference(p, x0)
            deb.append((p, x0, p.objective(x0), p.objective(u)))
    clf = []
    for seed in CLF_SEEDS:
        p = make_logistic(seed)
        T = p.T_step(0.995 / p.L)
        w0 = np.zeros(p.A.shape[1])
        clf.append((T, w0, float(np.linalg.norm(w0 - T(w0)))))
    out, rows = {}, []
    for name, mu, rho in VERSIONS:
        s_sfp, s_comp = variant(SFP_SETTING, mu, rho), variant(COMP_SETTING, mu, rho)
        a = [evals_to_reach(lambda N, cb, x0=x0: run_dispsa(sfp.T, x0, N, SFP_BETA, s_sfp, cb), EVALS_DISPSA,
                            SFP_MAX_EVALS, lambda x: sfp.rel_error(x) <= SFP_TOL) for _, _, x0 in sfp.starts]
        b = [evals_to_reach(lambda N, cb, p=p, x0=x0: run_dispsa(p.T, x0, N, COMP_BETA, s_comp, cb), EVALS_DISPSA,
                            DEBLUR_MAX, lambda c, p=p, F0=F0, Fr=Fr: (p.objective(c) - Fr) / (F0 - Fr) <= DEBLUR_GAP_TOL)
             for p, x0, F0, Fr in deb]
        c = [evals_to_reach(lambda N, cb, T=T, w0=w0: run_dispsa(T, w0, N, COMP_BETA, s_comp, cb), EVALS_DISPSA,
                            CLF_MAX_EVALS, lambda w, T=T, r0=r0: np.linalg.norm(w - T(w)) <= CLF_RES_FACTOR * r0)
             for T, w0, r0 in clf]
        out[name] = {"mu": s_comp.mu, "rho": s_comp.rho, "mu_sfp": s_sfp.mu, "rho_sfp": s_sfp.rho,
                     "sfp": a, "deblurring": b, "classification": c,
                     "means": [float(np.mean(a)), float(np.mean(b)), float(np.mean(c))]}
        rows.append(f"{ROW_LABELS[name]} & {np.mean(a):.1f} & {np.mean(b):.1f} & {np.mean(c):.1f} \\\\")
        print(name, out[name]["means"], flush=True)
    save_table(root, "table8_ablation.tex", rows)
    save_json(root, "ablation_results.json", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
