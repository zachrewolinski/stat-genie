import csv
import math
from typing import Dict, List, Tuple

import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normal_cdf(x: np.ndarray) -> np.ndarray:
    # Standard normal CDF via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fit_binomial_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a binomial logistic regression using Newton-Raphson."""
    N, K = X.shape
    beta = np.zeros(K)

    for _ in range(max_iter):
        eta = X @ beta
        p = logistic(eta)
        p = np.clip(p, 1e-9, 1 - 1e-9)

        # Gradient and observed information (negative Hessian)
        resid = y - n * p
        g = X.T @ resid
        w = n * p * (1.0 - p)
        WX = X * w[:, None]
        H_neg = X.T @ WX

        try:
            delta = np.linalg.solve(H_neg, g)
        except np.linalg.LinAlgError:
            H_neg_reg = H_neg + 1e-6 * np.eye(K)
            delta = np.linalg.solve(H_neg_reg, g)

        beta_new = beta + delta
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final covariance, standard errors, z-scores, and p-values
    eta = X @ beta
    p = logistic(eta)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    w = n * p * (1.0 - p)
    WX = X * w[:, None]
    H_neg = X.T @ WX

    try:
        cov = np.linalg.inv(H_neg)
    except np.linalg.LinAlgError:
        cov = np.full((K, K), np.nan)

    se = np.sqrt(np.diag(cov))
    z = beta / se
    # Vectorized normal CDF
    vec_normal_cdf = np.vectorize(normal_cdf)
    p_values = 2.0 * (1.0 - vec_normal_cdf(np.abs(z)))

    return beta, se, z, p_values, cov


def main() -> None:
    rows: List[Dict[str, str]] = []
    with open("amtl.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sockets = int(row["feature4"])
            if sockets <= 0:
                continue
            rows.append(row)

    genera = sorted({r["feature8"] for r in rows})
    if "Homo sapiens" in genera:
        genera.remove("Homo sapiens")
        genera.insert(0, "Homo sapiens")

    tooth_classes = sorted({r["feature1"] for r in rows})
    if "Anterior" in tooth_classes:
        tooth_classes.remove("Anterior")
        tooth_classes.insert(0, "Anterior")

    # Aggregate AMTL by genus
    genus_counts: Dict[str, Tuple[int, int]] = {g: (0, 0) for g in genera}
    for r in rows:
        g = r["feature8"]
        missing = int(r["feature3"])
        sockets = int(r["feature4"])
        m_sum, s_sum = genus_counts[g]
        genus_counts[g] = (m_sum + missing, s_sum + sockets)

    print("Weighted AMTL (missing / observable sockets) by genus:")
    for g in genera:
        missing, sockets = genus_counts[g]
        rate = missing / sockets if sockets > 0 else float("nan")
        print(f"{g:12s}  missing={missing:4d}, sockets={sockets:5d}, rate={rate:.4f}")
    print()

    # Build design matrix for logistic regression
    num_genus_dummy = len(genera) - 1  # baseline: Homo sapiens
    num_tooth_dummy = len(tooth_classes) - 1  # baseline: Anterior

    N = len(rows)
    K = 1 + num_genus_dummy + num_tooth_dummy + 2  # intercept + genus + tooth + age + sex
    X = np.zeros((N, K), dtype=float)
    y = np.zeros(N, dtype=float)
    n = np.zeros(N, dtype=float)

    genus_offset = 1
    tooth_offset = genus_offset + num_genus_dummy
    age_col = tooth_offset + num_tooth_dummy
    sex_col = age_col + 1

    for i, r in enumerate(rows):
        genus = r["feature8"]
        tooth = r["feature1"]
        missing = int(r["feature3"])
        sockets = int(r["feature4"])
        age = float(r["feature5"])
        sex = float(r["feature7"])

        X[i, 0] = 1.0  # intercept

        # Genus dummies (baseline: Homo sapiens)
        g_idx = genera.index(genus)
        if g_idx > 0:
            X[i, genus_offset + g_idx - 1] = 1.0

        # Tooth-class dummies (baseline: Anterior)
        t_idx = tooth_classes.index(tooth)
        if t_idx > 0:
            X[i, tooth_offset + t_idx - 1] = 1.0

        X[i, age_col] = age
        X[i, sex_col] = sex

        y[i] = missing
        n[i] = sockets

    beta, se, z, p_values, _ = fit_binomial_logistic(X, y, n)

    param_names: List[str] = ["Intercept"]
    param_names.extend(f"genus={g}" for g in genera[1:])
    param_names.extend(f"tooth={t}" for t in tooth_classes[1:])
    param_names.extend(["age", "sex"])

    print("Binomial logistic regression results")
    print("Outcome: number of missing teeth (of given class) out of observable sockets")
    print(
        "Predictors: genus (baseline Homo sapiens), age at death, sex estimate, tooth class (baseline Anterior)"
    )
    print()
    print(
        f"{'Parameter':20s} {'Coef':>10s} {'SE':>10s} {'z':>10s} {'p-value':>12s} {'OddsRatio':>12s}"
    )
    for name, b, s, zval, pval in zip(param_names, beta, se, z, p_values):
        or_val = math.exp(b)
        print(
            f"{name:20s} {b:10.4f} {s:10.4f} {zval:10.3f} {pval:12.4g} {or_val:12.4f}"
        )


if __name__ == "__main__":
    main()

