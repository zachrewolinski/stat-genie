import csv
import json
import math
from pathlib import Path
from typing import List, Tuple


def logistic(z: float) -> float:
    """Numerically stable logistic."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def normal_cdf(z: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def invert_3x3(mat: List[List[float]]) -> List[List[float]]:
    """Invert a 3x3 matrix using Gauss-Jordan elimination."""
    n = 3
    # Build augmented matrix [mat | I]
    aug = [
        mat[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # Pivot selection
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot_val = aug[pivot_row][col]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")

        # Swap rows if needed
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot_val = aug[col][col]
        inv_pivot = 1.0 / pivot_val
        for j in range(2 * n):
            aug[col][j] *= inv_pivot

        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]

    inv = [[aug[i][j] for j in range(n, 2 * n)] for i in range(n)]
    return inv


def fit_logistic_regression(
    y: List[int],
    x1: List[float],
    x2: List[float],
    max_iter: int = 20000,
    lr: float = 0.01,
    tol: float = 1e-6,
) -> Tuple[List[float], List[List[float]]]:
    """Fit logistic regression using gradient ascent and return (beta, covariance)."""
    n = len(y)
    if not (len(x1) == len(x2) == n):
        raise ValueError("Mismatched lengths in data vectors")

    # Design matrix columns: [1, x1, x2]
    beta = [0.0, 0.0, 0.0]

    for _ in range(max_iter):
        grad = [0.0, 0.0, 0.0]
        for i in range(n):
            eta = beta[0] + beta[1] * x1[i] + beta[2] * x2[i]
            p = logistic(eta)
            diff = y[i] - p
            grad[0] += diff
            grad[1] += diff * x1[i]
            grad[2] += diff * x2[i]

        max_change = 0.0
        for j in range(3):
            delta = lr * grad[j] / float(n)
            beta[j] += delta
            if abs(delta) > max_change:
                max_change = abs(delta)

        if max_change < tol:
            break

    # Compute observed Fisher information matrix: X^T W X
    info = [[0.0] * 3 for _ in range(3)]
    for i in range(n):
        eta = beta[0] + beta[1] * x1[i] + beta[2] * x2[i]
        p = logistic(eta)
        w = p * (1.0 - p)
        x_vec = [1.0, x1[i], x2[i]]
        for j in range(3):
            for k in range(3):
                info[j][k] += w * x_vec[j] * x_vec[k]

    cov = invert_3x3(info)
    return beta, cov


def standardize(xs: List[float]) -> Tuple[List[float], float, float]:
    n = len(xs)
    mean = sum(xs) / float(n)
    var = sum((x - mean) ** 2 for x in xs) / float(max(n - 1, 1))
    std = math.sqrt(var) if var > 0.0 else 1.0
    zs = [(x - mean) / std for x in xs]
    return zs, mean, std


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (not used programmatically beyond documentation)
    info = json.loads((base_path / "info.json").read_text())
    _ = info  # silence unused-variable warning

    # According to info.json descriptions (despite shuffled column names):
    # - m_focal: 1 if focal won contest, 0 otherwise  -> outcome
    # - f_other: number of individuals in focal group -> focal group size
    # - win:     number of individuals in other group -> other group size
    # - m_other: distance of focal group from its home-range center
    # - n_focal: distance of other group from its home-range center

    y: List[int] = []
    rel_size_raw: List[float] = []
    rel_loc_raw: List[float] = []

    with (base_path / "crofoot.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_i = int(row["m_focal"])
            focal_size = float(row["f_other"])
            other_size = float(row["win"])
            focal_dist = float(row["m_other"])
            other_dist = float(row["n_focal"])

            y.append(y_i)
            # positive -> focal group larger
            rel_size_raw.append(focal_size - other_size)
            # positive -> contest closer to focal group's home range
            rel_loc_raw.append(other_dist - focal_dist)

    # Standardize predictors for numerical stability
    rel_size, rel_size_mean, rel_size_std = standardize(rel_size_raw)
    rel_loc, rel_loc_mean, rel_loc_std = standardize(rel_loc_raw)

    beta, cov = fit_logistic_regression(y, rel_size, rel_loc)

    # Extract coefficients and standard errors
    se = [math.sqrt(max(cov[j][j], 0.0)) for j in range(3)]
    z_stats = [
        (beta[j] / se[j]) if se[j] > 0.0 else float("nan") for j in range(3)
    ]
    p_values = [
        2.0 * (1.0 - normal_cdf(abs(z))) if not math.isnan(z) else float("nan")
        for z in z_stats
    ]

    # Coefficients correspond to 1-standard-deviation changes in predictors
    coef_intercept = beta[0]
    coef_rel_size = beta[1]
    coef_rel_loc = beta[2]

    p_rel_size = p_values[1]
    p_rel_loc = p_values[2]

    # Odds ratios for a 1-SD increase in each predictor
    or_rel_size = math.exp(coef_rel_size)
    or_rel_loc = math.exp(coef_rel_loc)
    or_intercept = math.exp(coef_intercept)

    best_p = min(p_rel_size, p_rel_loc)

    both_sig = (p_rel_size < 0.05) and (p_rel_loc < 0.05)
    at_least_one_sig = (p_rel_size < 0.05) or (p_rel_loc < 0.05)
    both_in_theoretical_direction = (coef_rel_size > 0.0) and (coef_rel_loc > 0.0)

    if both_sig and both_in_theoretical_direction:
        response = 90
    elif at_least_one_sig and both_in_theoretical_direction:
        response = 75
    elif best_p < 0.1 and (coef_rel_size > 0.0 or coef_rel_loc > 0.0):
        response = 60
    elif best_p < 0.2:
        response = 40
    else:
        response = 20

    # Build human-readable explanation
    lines: List[str] = []
    lines.append(
        "Research question: Do relative group size and contest location "
        "influence the probability of a capuchin monkey group winning an intergroup contest?"
    )

    lines.append(
        "Outcome variable: binary indicator of whether the focal group won the contest."
    )
    lines.append(
        "Predictors: (1) relative group size (focal group size minus other group size), "
        "(2) relative contest location (other group's distance from its home-range center "
        "minus focal group's distance, so positive values indicate the contest occurs "
        "closer to the focal group's home range). Both predictors were standardized "
        "to have mean 0 and unit variance before fitting the model."
    )

    lines.append(
        "I fit a logistic regression (maximum likelihood with gradient ascent) "
        "of focal win on these two predictors."
    )

    lines.append(
        f"Relative group size coefficient (per 1 SD increase): {coef_rel_size:.3f} "
        f"(odds ratio {or_rel_size:.2f}, p-value {p_rel_size:.3g})."
    )
    lines.append(
        f"Relative location coefficient (per 1 SD increase): {coef_rel_loc:.3f} "
        f"(odds ratio {or_rel_loc:.2f}, p-value {p_rel_loc:.3g})."
    )

    lines.append(
        "Here, a 1 SD increase in relative group size means that the focal group's "
        "size advantage over its opponent increases by one standard deviation; "
        "a 1 SD increase in relative location means the contest is substantially "
        "closer to the focal group's home-range center compared to the other group's."
    )

    lines.append(
        f"Model intercept (baseline log-odds of winning when predictors are at their means): "
        f"{coef_intercept:.3f} (odds ratio {or_intercept:.2f})."
    )

    if response >= 75:
        qualitative = (
            "The regression results provide clear evidence that these factors "
            "influence the probability of winning: at least one predictor is "
            "statistically significant at the 5% level with an effect in the "
            "expected direction, and the estimated odds ratios indicate that "
            "larger relative group size and/or being closer to the home range "
            "meaningfully increase the chance of winning."
        )
    elif response >= 60:
        qualitative = (
            "The regression results provide moderate evidence that these factors "
            "influence the probability of winning: effect estimates are in the "
            "expected direction and approach conventional significance thresholds, "
            "but the limited sample size means uncertainty remains."
        )
    elif response >= 40:
        qualitative = (
            "The regression results provide weak or ambiguous evidence that these "
            "factors influence the probability of winning: effect estimates are in "
            "theoretically plausible directions but do not robustly reach standard "
            "significance thresholds, so the data do not allow strong conclusions."
        )
    else:
        qualitative = (
            "The regression results do not provide evidence that relative group size "
            "or contest location influence the probability of winning: neither predictor "
            "approaches conventional significance thresholds, and effect estimates are "
            "small relative to their uncertainty."
        )

    lines.append(qualitative)

    explanation = "\n".join(lines)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    (base_path / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

