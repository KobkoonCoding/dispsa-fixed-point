"""The fixed point algorithms compared in the paper.

* ``run_bot``    : the Tikhonov-regularized Krasnosel'skii-Mann iteration of Bot and Meier (2021),
                       x_{n+1} = beta_n x_n + lambda_n (T_n(beta_n x_n) - beta_n x_n),
                   with one evaluation of T_n per iteration.  For T_n = T it is the scheme of
                   Bot, Csetnek and Meier (2019); for T_n = prox_{r_n f}(I - r_n grad g) with
                   variable step sizes it is the forward-backward algorithm of Bot and Meier
                   (Section 4.3).
* ``run_dispsa`` : Algorithm 1 (DISPSA) of the paper,
                       theta_n = min{mu_n, eps_n/||x_n-x_{n-1}||},
                       delta_n = max{rho_n, -eps_n/||x_{n-1}-x_{n-2}||},
                       y_n = x_n + theta_n (x_n-x_{n-1}) + delta_n (x_{n-1}-x_{n-2}),
                       z_n = (1-alpha_n) beta_n y_n + alpha_n T_n(beta_n y_n),
                       x_{n+1} = (1-lambda_n) T_n y_n + lambda_n T_n z_n,
                   with three evaluations of T_n per iteration.

``beta`` and ``eps`` are callables n -> beta_n and n -> epsilon_n (n starts at 1).
``callback(n, x)`` is called after every iteration with the new iterate; if it returns True the
iteration stops.  ``Tn`` (optional) is a callable n -> T_n; otherwise T_n = T for all n.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EVALS_BOT = 1      # evaluations of T_n per iteration of the method of Bot et al.
EVALS_DISPSA = 3   # evaluations of T_n per iteration of DISPSA


@dataclass(frozen=True)
class DispsaSetting:
    """Constant control parameters alpha_n, lambda_n, mu_n, rho_n and epsilon_n = c_eps (1-beta_n)^p_eps."""

    alpha: float
    lam: float
    mu: float
    rho: float
    c_eps: float
    p_eps: float

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha_n must lie in (0,1)")
        if not 0.0 < self.lam <= 1.0:
            raise ValueError("lambda_n must lie in (0,1]")
        if self.mu < 0.0 or self.rho > 0.0:
            raise ValueError("Algorithm 1 requires mu_n >= 0 and rho_n <= 0")
        if self.c_eps <= 0.0 or self.p_eps <= 1.0:
            raise ValueError("epsilon_n/(1-beta_n) -> 0 needs c_eps > 0 and p_eps > 1")


def power_beta(c: float, q: float) -> Callable[[int], float]:
    """beta_n = 1 - c/(n+2)^q, n >= 1; for 0 < q <= 1 and 0 < c < 3^q it satisfies condition (i)."""
    if not (0.0 < q <= 1.0 and 0.0 < c < 3.0 ** q):
        raise ValueError("need 0 < q <= 1 and 0 < c < 3^q")
    return lambda n: 1.0 - c / (n + 2.0) ** q


def _family(T, Tn):
    return Tn if Tn is not None else (lambda n: T)


def run_bot(T, x0, N, beta, lam, callback=None, Tn=None):
    """N iterations of the method of Bot et al. (Bot and Meier for a nonconstant family)."""
    fam = _family(T, Tn)
    x = np.array(x0, dtype=float, copy=True)
    for n in range(1, N + 1):
        b = beta(n)
        bx = b * x
        x = bx + lam * (fam(n)(bx) - bx)
        if callback is not None and callback(n, x):
            return x, n
    return x, N


def run_dispsa(T, x0, N, beta, setting: DispsaSetting, callback=None, Tn=None, norm=np.linalg.norm):
    """N iterations of Algorithm 1 with x_{-1} = x_0 = x_1 = x0."""
    fam = _family(T, Tn)
    s = setting
    x_nm2 = np.array(x0, dtype=float, copy=True)   # x_{n-2}
    x_nm1 = x_nm2.copy()                           # x_{n-1}
    x = x_nm2.copy()                               # x_n
    for n in range(1, N + 1):
        b = beta(n)
        e = s.c_eps * (1.0 - b) ** s.p_eps
        d1, d2 = x - x_nm1, x_nm1 - x_nm2
        n1, n2 = norm(d1), norm(d2)
        theta = min(s.mu, e / n1) if n1 > 0 else s.mu          # (9)
        delta = max(s.rho, -e / n2) if n2 > 0 else s.rho       # (10)
        y = x + theta * d1 + delta * d2                        # (11)
        Tcur = fam(n)
        by = b * y
        z = (1.0 - s.alpha) * by + s.alpha * Tcur(by)          # (12)
        x_new = (1.0 - s.lam) * Tcur(y) + s.lam * Tcur(z)      # (13)
        x_nm2, x_nm1, x = x_nm1, x, x_new
        if callback is not None and callback(n, x):
            return x, n
    return x, N


def evals_to_reach(run, evals_per_iter, max_evals, test):
    """Run ``run(N, callback)`` and return the number of evaluations of T_n after which
    ``test(x)`` first holds (None if it does not hold within ``max_evals``)."""
    hit = {}

    def cb(n, x):
        if test(x):
            hit["n"] = n
            return True
        return False

    run(max_evals // evals_per_iter, cb)
    return hit["n"] * evals_per_iter if "n" in hit else None
