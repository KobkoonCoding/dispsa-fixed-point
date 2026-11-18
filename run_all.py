"""Reproduce every table and figure of Section 4 of the paper.  Usage:  python run_all.py [results_dir]"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

from dispsa import ablation, classification, deblurring, select_parameters, sfp, varstep  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "results"
STEPS = [("Table 1: selection of the parameters", select_parameters),
         ("Section 4.1: Table 3", sfp),
         ("Section 4.2: Tables 4-5, Figure 1", deblurring),
         ("Section 4.3: Table 6, Table 7 (first row)", classification),
         ("Section 4.3: Table 7 (second row, variable step sizes)", varstep),
         ("Section 4.4: Table 8", ablation)]
for name, mod in STEPS:
    t0 = time.time()
    print(f"=== {name} ===", flush=True)
    mod.main(out)
    print(f"--- done in {time.time() - t0:.0f} s\n", flush=True)
print(f"All tables ({out}/tables/*.tex) and the figure ({out}/figures/*.pdf) written.")
