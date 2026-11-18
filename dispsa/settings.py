"""Parameter settings of Section 4 (fixed before the reported runs; see README.md)."""
from .algorithms import DispsaSetting, power_beta

# Section 4.1 (split feasibility problem)
SFP_BETA = power_beta(0.90, 0.80)                       # beta_n = 1 - 0.9/(n+2)^0.8
SFP_SETTING = DispsaSetting(alpha=0.95, lam=0.98, mu=1.0, rho=-0.20, c_eps=10.0, p_eps=2.0)

# Sections 4.2-4.4 (deblurring and classification)
COMP_BETA = power_beta(0.30, 0.55)                      # beta_n = 1 - 0.3/(n+2)^0.55
COMP_SETTING = DispsaSetting(alpha=0.95, lam=0.30, mu=1.0, rho=-0.03, c_eps=50.0, p_eps=2.3)

# Relaxation parameters of the method of Bot et al. (all admissible, see Section 4)
BOT_GRID = (0.4, 1.0, 1.4)

SFP_SEED = 20260903
SFP_TOL = 1.0e-4
SFP_MAX_EVALS = 150_000

DEBLUR_NOISE = (0.005, 0.010, 0.015, 0.020, 0.025)
DEBLUR_SEEDS = (20260903, 20260904, 20260905, 20260906)
DEBLUR_BUDGET = 60
DEBLUR_GAP_TOL = 1.0e-2
DEBLUR_MAX_EVALS = 3000
DEBLUR_REF_ITERS = 5000

CLF_SEEDS = tuple(range(20260903, 20260933))
CLF_BUDGET = 900
CLF_RES_FACTOR = 0.1
CLF_MAX_EVALS = 3000
