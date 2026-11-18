# dispsa-fixed-point

Code, data and stored results for the paper

> Suthep Suantai, Bancha Panyanak, and Kobkoon Janngam, *A two-step inertial S-iteration algorithm with Tikhonov
> regularization for common fixed points and forward-backward splitting, with applications to image deblurring and
> data classification*.

The paper proposes DISPSA (Algorithm 1) and compares it with the Tikhonov-regularized Krasnosel'skii-Mann algorithm of
Boţ and Meier (2021). Both algorithms are implemented in `dispsa/algorithms.py`.

## What this repository reproduces

| Item in the paper | Script | Output (in `results/`) |
|---|---|---|
| Table 1 (selection of alpha_n, lambda_n, rho_n and of lambda_l1) | `dispsa/select_parameters.py` | `parameter_selection.json` |
| Table 3 and the error values of Section 4.1 | `dispsa/sfp.py` | `tables/table3_sfp.tex`, `sfp_results.json`, `sfp_curves.json` |
| Tables 4 and 5, Figure 1 | `dispsa/deblurring.py` | `tables/table4_deblur_quality.tex`, `tables/table5_deblur_evaluations.tex`, `deblur_results.json`, `figures/fig1_restored_images.pdf` |
| Table 6, first row of Table 7 | `dispsa/classification.py` | `tables/table6_classification.tex`, `tables/table7_row1_constant_step.tex`, `classification_results.json` |
| Second row of Table 7 (variable step sizes) | `dispsa/varstep.py` | `tables/table7_row2_variable_step.tex`, `varstep_results.json` |
| Table 8 (ablation of the inertial terms) | `dispsa/ablation.py` | `tables/table8_ablation.tex`, `ablation_results.json` |

Table 2 of the paper lists the settings; they are in `dispsa/settings.py` and `dispsa/problems.py`.
The folder `expected_results/` holds the outputs used in the paper.

## Quick start

Run the commands from the repository root.

```
pip install -r requirements.txt
python run_all.py results
python verify.py results expected_results
```

`run_all.py` takes about 15 minutes on one core of a laptop. The last command prints
`All tables and numbers agree with the expected results.` if every table file is identical and every number in the
JSON files agrees to a relative tolerance of 1e-6. Each experiment can also be run alone, for example
`python -m dispsa.sfp results`.

The results of the paper were computed with Python 3.13.5 on Windows 11 and the package versions in
`requirements.txt`. The figure is compared by eye: its pixels can differ slightly with other fonts or versions of
Matplotlib.

## Algorithms

```
theta_n = min(mu_n, eps_n/||x_n - x_{n-1}||),   delta_n = max(rho_n, -eps_n/||x_{n-1} - x_{n-2}||)
y_n     = x_n + theta_n (x_n - x_{n-1}) + delta_n (x_{n-1} - x_{n-2})
z_n     = (1 - alpha_n) beta_n y_n + alpha_n T_n(beta_n y_n)
x_{n+1} = (1 - lambda_n) T_n(y_n) + lambda_n T_n(z_n)                        (three evaluations of T_n)

Bot and Meier:  x_{n+1} = beta_n x_n + lambda_n (T_n(beta_n x_n) - beta_n x_n)   (one evaluation of T_n)
```

DISPSA starts with x_{-1} = x_0 = x_1. The cost of a run is the number of evaluations of T (or of T_n), so it does not
depend on the machine. Evaluations used only in a stopping test are not counted.

## Data

- `data/wdbc.csv`: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository,
  https://doi.org/10.24432/C5DW2B, license CC BY 4.0. Column `target`: 0 = malignant (the positive class).
- The camera image is loaded from scikit-image (`skimage.data.camera()`, CC0); it is not stored here.
- The data of the split feasibility problem are generated from the seed 20260903.

## License

The code is released under the MIT License (`LICENSE`). The dataset keeps its own license (see Data).
