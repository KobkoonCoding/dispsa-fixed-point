"""Section 4.2: wavelet-based image deblurring; Tables 4-5 and Figure 1."""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .algorithms import EVALS_BOT, EVALS_DISPSA, evals_to_reach, run_bot, run_dispsa
from .common import bot_key, outdir, save_json, save_table
from .problems import haar2, make_deblur
from .settings import (BOT_GRID, COMP_BETA, COMP_SETTING, DEBLUR_BUDGET, DEBLUR_GAP_TOL, DEBLUR_MAX_EVALS,
                       DEBLUR_NOISE, DEBLUR_REF_ITERS, DEBLUR_SEEDS)

MAX_EVALS = DEBLUR_MAX_EVALS
SHOW_CASE = (0.010, DEBLUR_SEEDS[0])


def psnr(p, img) -> float:
    return float(peak_signal_noise_ratio(p.truth, img, data_range=1.0))


def ssim(p, img) -> float:
    return float(structural_similarity(p.truth, img, data_range=1.0))


def reference(p, x0):
    x = x0.copy()
    for _ in range(DEBLUR_REF_ITERS):
        x = p.T(x)
    return x


def one_case(sigma: float, seed: int) -> dict:
    p = make_deblur(sigma, seed)
    x0 = haar2(p.observed, p.levels).ravel()          # x_0 = W b, the coefficients of the observed image
    u = reference(p, x0)
    F0, Fr = p.objective(x0), p.objective(u)
    hit = lambda c: (p.objective(c) - Fr) / (F0 - Fr) <= DEBLUR_GAP_TOL
    row = {"sigma": sigma, "seed": seed, "F_ref": Fr, "F_0": F0,
           "ref_residual_ratio": float(np.linalg.norm(u - p.T(u)) / np.linalg.norm(x0 - p.T(x0))),
           "psnr_b": psnr(p, p.observed), "ssim_b": ssim(p, p.observed),
           "psnr_ref": psnr(p, p.image(u)), "ssim_ref": ssim(p, p.image(u))}
    row["evals DISPSA"] = evals_to_reach(lambda N, cb: run_dispsa(p.T, x0, N, COMP_BETA, COMP_SETTING, cb),
                                         EVALS_DISPSA, MAX_EVALS, hit)
    for lam in BOT_GRID:
        row[f"evals {bot_key(lam)}"] = evals_to_reach(
            lambda N, cb, lam=lam: run_bot(p.T, x0, N, COMP_BETA, lam, cb), EVALS_BOT, MAX_EVALS, hit)
    xd, _ = run_dispsa(p.T, x0, DEBLUR_BUDGET // EVALS_DISPSA, COMP_BETA, COMP_SETTING)
    row["psnr DISPSA"], row["ssim DISPSA"] = psnr(p, p.image(xd)), ssim(p, p.image(xd))
    for lam in BOT_GRID:
        xb, _ = run_bot(p.T, x0, DEBLUR_BUDGET, COMP_BETA, lam)
        row[f"psnr {bot_key(lam)}"], row[f"ssim {bot_key(lam)}"] = psnr(p, p.image(xb)), ssim(p, p.image(xb))
    return row


def restored_images(sigma: float, seed: int) -> dict:
    """The images of Figure 1: original, observed, and the restorations after DEBLUR_BUDGET evaluations of T."""
    p = make_deblur(sigma, seed)
    x0 = haar2(p.observed, p.levels).ravel()
    imgs = {"original": p.truth, "observed": p.observed}
    xd, _ = run_dispsa(p.T, x0, DEBLUR_BUDGET // EVALS_DISPSA, COMP_BETA, COMP_SETTING)
    imgs["DISPSA"] = p.image(xd)
    for lam in (1.0, 1.4):
        xb, _ = run_bot(p.T, x0, DEBLUR_BUDGET, COMP_BETA, lam)
        imgs[bot_key(lam)] = p.image(xb)
    return imgs


def plot_images(imgs: dict, root) -> None:
    order = [("original", "Original"), ("observed", "Observed"), ("DISPSA", "DISPSA"),
             ("Bot (1)", r"Boţ et al., $\lambda_n=1$"), ("Bot (1.4)", r"Boţ et al., $\lambda_n=1.4$")]
    truth = imgs["original"]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    for ax, (key, title) in zip(axes, order):
        ax.imshow(imgs[key], cmap="gray", vmin=0, vmax=1)
        sub = "" if key == "original" else (f"\n{peak_signal_noise_ratio(truth, imgs[key], data_range=1.0):.2f} dB, "
                                            f"{structural_similarity(truth, imgs[key], data_range=1.0):.4f}")
        ax.set_title(title + sub, fontsize=7.5)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir(root, "figures") / "fig1_restored_images.pdf", bbox_inches="tight", pad_inches=0.03,
                metadata={"CreationDate": None})
    plt.close(fig)


def table_rows(rows: list[dict]) -> tuple[list[str], list[str]]:
    t2, t3 = [], []
    for sigma in DEBLUR_NOISE:
        rs = [r for r in rows if r["sigma"] == sigma]
        mean = lambda k: float(np.mean([r[k] for r in rs]))
        cells = []
        for key in ("DISPSA", "Bot (1)", "Bot (1.4)"):
            v = [r[f"evals {key}"] for r in rs]
            if None in v:
                cells.append(f"$>{DEBLUR_MAX_EVALS}$")
            else:
                cells.append(f"{np.mean(v):.1f} ({min(v)}--{max(v)})")
        t3.append(f"{sigma:.3f} & " + " & ".join(cells) + " \\\\")
        t2.append(f"{sigma:.3f} & {mean('psnr_b'):.2f} & {mean('psnr DISPSA'):.2f} & {mean('psnr Bot (1)'):.2f} & "
                  f"{mean('psnr Bot (1.4)'):.2f} & {mean('psnr_ref'):.2f} & {mean('ssim DISPSA'):.4f} & "
                  f"{mean('ssim Bot (1)'):.4f} & {mean('ssim Bot (1.4)'):.4f} & {mean('ssim_ref'):.4f} \\\\")
    return t2, t3


def main(root: str = "results") -> None:
    rows = []
    for sigma in DEBLUR_NOISE:
        for seed in DEBLUR_SEEDS:
            rows.append(one_case(sigma, seed))
            print(f"deblurring sigma={sigma} seed={seed} done", flush=True)
    t2, t3 = table_rows(rows)
    save_table(root, "table4_deblur_quality.tex", t2)
    save_table(root, "table5_deblur_evaluations.tex", t3)
    save_json(root, "deblur_results.json", {"cases": rows})
    plot_images(restored_images(*SHOW_CASE), root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
