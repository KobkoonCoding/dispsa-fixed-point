"""Selection of the parameters (Table 1 and Section 4 of the paper) on development problems generated with the seed
20260831, which is not used in the reported runs.

* lambda_l1 of the deblurring problem: grid {0.0005, 0.001, 0.002, 0.004, 0.008, 0.016}; criterion: largest PSNR of
  the reference solution (5000 forward-backward iterations from W b) for sigma_noise = 0.010.
* lambda_l1 of the classification problem: grid {0.000625, 0.00125, 0.0025, 0.005, 0.01}; criterion: largest test
  accuracy of the reference solution (5000 forward-backward iterations from 0); ties are broken by the larger value.
* Settings (S1) and (S2): grid alpha in {0.75, 0.95}, lambda in {0.3, 0.98}, rho in {-0.03, -0.2}, with mu = 1 and the
  sequences beta_n, epsilon_n of (S1) and (S2).  Criterion for (S1): total evaluations of T to satisfy (47) over the five
  initial points of the split feasibility problem.  Criterion for (S2): evaluations to satisfy (52) for the deblurring
  problem plus evaluations to satisfy (56) for the classification problem.
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
from sklearn.metrics import accuracy_score

from .algorithms import EVALS_DISPSA, evals_to_reach, run_dispsa
from .common import save_json
from .deblurring import psnr
from .problems import haar2, make_affine_sfp, make_deblur, make_logistic
from .settings import (CLF_MAX_EVALS, CLF_RES_FACTOR, COMP_BETA, COMP_SETTING, DEBLUR_GAP_TOL, DEBLUR_MAX_EVALS,
                       DEBLUR_REF_ITERS, SFP_BETA, SFP_MAX_EVALS, SFP_SETTING, SFP_TOL)

DEV_SEED = 20260831
GRID = [(a, lam, rho) for a in (0.75, 0.95) for lam in (0.3, 0.98) for rho in (-0.03, -0.2)]
DEBLUR_LAMS = (0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)
CLF_LAMS = (0.000625, 0.00125, 0.0025, 0.005, 0.01)


def _cap(v, cap):
    return cap if v is None else v


def select_deblur_lambda() -> dict:
    out = {}
    for lam in DEBLUR_LAMS:
        p = make_deblur(0.010, DEV_SEED, tau=lam)
        x = haar2(p.observed, p.levels).ravel()
        for _ in range(DEBLUR_REF_ITERS):
            x = p.T(x)
        out[str(lam)] = psnr(p, p.image(x))
    return out


def select_clf_lambda() -> dict:
    out = {}
    for lam in CLF_LAMS:
        p = make_logistic(DEV_SEED, tau=lam)
        T = p.T_step(0.995 / p.L)
        w = np.zeros(p.A.shape[1])
        for _ in range(5000):
            w = T(w)
        out[str(lam)] = 100.0 * accuracy_score(p.y_test, (p.A_test @ w > 0).astype(int))
    return out


def select_s1() -> dict:
    p = make_affine_sfp(DEV_SEED)
    out = {}
    for a, lam, rho in GRID:
        s = replace(SFP_SETTING, alpha=a, lam=lam, rho=rho)
        c = [evals_to_reach(lambda N, cb, x0=x0: run_dispsa(p.T, x0, N, SFP_BETA, s, cb), EVALS_DISPSA,
                            SFP_MAX_EVALS, lambda x: p.rel_error(x) <= SFP_TOL) for _, _, x0 in p.starts]
        out[f"alpha={a},lambda={lam},rho={rho}"] = sum(_cap(v, SFP_MAX_EVALS) for v in c)
    return out


def select_s2() -> dict:
    p = make_deblur(0.010, DEV_SEED)
    x0 = haar2(p.observed, p.levels).ravel()
    u = x0.copy()
    for _ in range(DEBLUR_REF_ITERS):
        u = p.T(u)
    F0, Fr = p.objective(x0), p.objective(u)
    q = make_logistic(DEV_SEED)
    T = q.T_step(0.995 / q.L)
    w0 = np.zeros(q.A.shape[1])
    r0 = float(np.linalg.norm(w0 - T(w0)))
    out = {}
    for a, lam, rho in GRID:
        s = replace(COMP_SETTING, alpha=a, lam=lam, rho=rho)
        d = evals_to_reach(lambda N, cb: run_dispsa(p.T, x0, N, COMP_BETA, s, cb), EVALS_DISPSA, DEBLUR_MAX_EVALS,
                           lambda c: (p.objective(c) - Fr) / (F0 - Fr) <= DEBLUR_GAP_TOL)
        c = evals_to_reach(lambda N, cb: run_dispsa(T, w0, N, COMP_BETA, s, cb), EVALS_DISPSA, CLF_MAX_EVALS,
                           lambda v: np.linalg.norm(v - T(v)) <= CLF_RES_FACTOR * r0)
        out[f"alpha={a},lambda={lam},rho={rho}"] = _cap(d, DEBLUR_MAX_EVALS) + _cap(c, CLF_MAX_EVALS)
    return out


def main(root: str = "results") -> None:
    res = {"deblur_lambda_psnr_of_reference": select_deblur_lambda(),
           "classification_lambda_test_accuracy": select_clf_lambda(),
           "S1_total_evaluations": select_s1(), "S2_sum_of_evaluations": select_s2()}
    save_json(root, "parameter_selection.json", res)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
