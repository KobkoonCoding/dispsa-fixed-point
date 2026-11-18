"""The three test problems of Section 4: affine split feasibility, wavelet deblurring, sparse logistic regression."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Array = np.ndarray

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- Section 4.1: split feasibility
@dataclass(frozen=True)
class AffineSFP:
    M: Array               # 42 x 240 with orthonormal rows
    d: Array
    x_dagger: Array        # solution with minimum norm, M^T d
    starts: tuple          # ((kappa, tau, x0), ...)
    step: float

    def T(self, x: Array) -> Array:
        return x - self.step * (self.M.T @ (self.M @ x - self.d))

    def rel_error(self, x: Array) -> float:
        return float(np.linalg.norm(x - self.x_dagger) / max(1.0, np.linalg.norm(self.x_dagger)))


SFP_STARTS = ((4, 1), (8, 3), (-6, 4), (12, -2), (-10, -5))


def make_affine_sfp(seed: int = 20260903, step: float = 0.99) -> AffineSFP:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((240, 42)), mode="reduced")
    M = q.T
    d = M @ rng.standard_normal(240)
    v1 = rng.standard_normal(240)
    v1 /= np.linalg.norm(v1)
    v2 = rng.standard_normal(240)
    v2 -= v1 * float(v1 @ v2)
    v2 /= np.linalg.norm(v2)
    starts = tuple((k, t, k * v1 + t * v2) for k, t in SFP_STARTS)
    return AffineSFP(M=M, d=d, x_dagger=M.T @ d, starts=starts, step=step)


# ---------------------------------------------------------------- Section 4.2: wavelet deblurring
def haar2(x: Array, levels: int) -> Array:
    a = np.asarray(x, dtype=float).copy()
    H, W = a.shape
    s = math.sqrt(2.0)
    for lev in range(levels):
        h, w = H >> lev, W >> lev
        blk = a[:h, :w].copy()
        tmp = np.concatenate([(blk[:, 0::2] + blk[:, 1::2]) / s, (blk[:, 0::2] - blk[:, 1::2]) / s], axis=1)
        a[:h, :w] = np.concatenate([(tmp[0::2] + tmp[1::2]) / s, (tmp[0::2] - tmp[1::2]) / s], axis=0)
    return a


def inverse_haar2(c: Array, levels: int) -> Array:
    a = np.asarray(c, dtype=float).copy()
    H, W = a.shape
    s = math.sqrt(2.0)
    for lev in range(levels - 1, -1, -1):
        h, w = H >> lev, W >> lev
        blk = a[:h, :w].copy()
        lo, hi = blk[: h // 2], blk[h // 2:h]
        tmp = np.empty((h, w))
        tmp[0::2], tmp[1::2] = (lo + hi) / s, (lo - hi) / s
        lo, hi = tmp[:, : w // 2], tmp[:, w // 2:w]
        out = np.empty((h, w))
        out[:, 0::2], out[:, 1::2] = (lo + hi) / s, (lo - hi) / s
        a[:h, :w] = out
    return a


@dataclass(frozen=True)
class Deblur:
    truth: Array
    observed: Array
    transfer: Array        # Fourier transfer function of the periodic Gaussian blur K
    tau: float             # weight of the l1 term
    nu: float              # weight of the ridge term
    levels: int
    step: float

    def K(self, img: Array) -> Array:
        return np.fft.irfft2(self.transfer * np.fft.rfft2(img), s=self.truth.shape)

    def Kt(self, img: Array) -> Array:
        return np.fft.irfft2(np.conj(self.transfer) * np.fft.rfft2(img), s=self.truth.shape)

    def objective(self, c: Array) -> float:
        c = c.reshape(self.truth.shape)
        r = self.K(inverse_haar2(c, self.levels)) - self.observed
        return 0.5 * float(np.sum(r * r)) + self.tau * float(np.abs(c).sum()) + 0.5 * self.nu * float(np.sum(c * c))

    def T(self, c: Array) -> Array:
        c = c.reshape(self.truth.shape)
        grad = haar2(self.Kt(self.K(inverse_haar2(c, self.levels)) - self.observed), self.levels) + self.nu * c
        v = c - self.step * grad
        return (np.sign(v) * np.maximum(np.abs(v) - self.step * self.tau, 0.0)).ravel()

    def image(self, c: Array) -> Array:
        return np.clip(inverse_haar2(c.reshape(self.truth.shape), self.levels), 0.0, 1.0)


def camera_128() -> Array:
    from skimage.data import camera
    from skimage.transform import resize
    img = resize(camera().astype(float) / 255.0, (128, 128), order=1, mode="reflect",
                 anti_aliasing=True, preserve_range=True)
    return np.clip(img, 0.0, 1.0)


def make_deblur(noise_sigma: float, noise_seed: int, tau: float = 1.0e-3, nu: float = 1.0e-4,
                blur_sigma: float = 1.6, levels: int = 4) -> Deblur:
    truth = camera_128()
    H, W = truth.shape
    yy, xx = np.meshgrid(np.minimum(np.arange(H), H - np.arange(H)).astype(float),
                         np.minimum(np.arange(W), W - np.arange(W)).astype(float), indexing="ij")
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * blur_sigma ** 2))
    transfer = np.fft.rfft2(kernel / kernel.sum())
    blurred = np.fft.irfft2(transfer * np.fft.rfft2(truth), s=truth.shape)
    rng = np.random.default_rng(noise_seed)
    observed = np.clip(blurred + noise_sigma * rng.standard_normal(truth.shape), 0.0, 1.0)
    # ||K|| = 1 and the Haar transform is orthonormal, so L = 1 + nu.
    return Deblur(truth=truth, observed=observed, transfer=transfer, tau=tau, nu=nu, levels=levels,
                  step=0.995 / (1.0 + nu))


# ---------------------------------------------------------------- Section 4.3: sparse logistic regression
@dataclass(frozen=True)
class Logistic:
    A: Array               # training samples with an appended intercept column (m x 31)
    b: Array               # training labels in {-1, 1}
    A_test: Array
    y_test: Array          # test labels in {0, 1}, 1 = malignant
    tau: float
    nu: float
    L: float               # Lipschitz constant of the gradient of g

    def grad(self, w: Array) -> Array:
        m = self.A.shape[0]
        return self.A.T @ (-self.b * expit(-self.b * (self.A @ w)) / m) + self.nu * w

    def objective(self, w: Array) -> float:
        loss = float(np.logaddexp(0.0, -self.b * (self.A @ w)).mean())
        return loss + self.tau * float(np.abs(w[:-1]).sum()) + 0.5 * self.nu * float(w @ w)

    def T_step(self, r: float) -> Callable[[Array], Array]:
        """The forward-backward mapping prox_{r f}(I - r grad g); the intercept is not thresholded."""
        def T(w: Array) -> Array:
            v = w - r * self.grad(w)
            out = v.copy()
            out[:-1] = np.sign(v[:-1]) * np.maximum(np.abs(v[:-1]) - r * self.tau, 0.0)
            return out
        return T


def make_logistic(split_seed: int, tau: float = 2.5e-3, nu: float = 1.0e-3) -> Logistic:
    frame = pd.read_csv(DATA / "wdbc.csv")
    y01 = (frame.pop("target").to_numpy(dtype=int) == 0).astype(int)   # malignant = positive class
    X = frame.to_numpy(dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y01, test_size=0.20, random_state=split_seed, stratify=y01)
    sc = StandardScaler().fit(Xtr)
    A = np.column_stack([sc.transform(Xtr), np.ones(len(Xtr))])
    A_test = np.column_stack([sc.transform(Xte), np.ones(len(Xte))])
    m = A.shape[0]
    L = float(np.linalg.eigvalsh(A.T @ A / (4.0 * m) + nu * np.eye(A.shape[1]))[-1])
    return Logistic(A=A, b=2.0 * ytr - 1.0, A_test=A_test, y_test=yte, tau=tau, nu=nu, L=L)
