import json
import math
from pathlib import Path

import numpy as np


def logistic(x: np.ndarray) -> np.ndarray:
    """Stable logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-x))


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fit_binomial_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a binomial logistic regression using IRLS.

    Returns
    -------
    beta : np.ndarray
        Estimated coefficients.
    XtWX : np.ndarray
        Final Fisher information matrix (X^T W X), used for standard errors.
    """
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        eta = X @ beta
        mu = logistic(eta)
        # Avoid probabilities at exactly 0 or 1
        mu = np.clip(mu, 1e-8, 1 - 1e-8)

        W = n_trials * mu * (1.0 - mu)

        # Working response
        y_prop = y / n_trials
        z = eta + (y_prop - mu) / (mu * (1.0 - mu))

        WX = X * W[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (W * z)

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            # Fall back to previous beta if matrix is singular
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    # Final information matrix for standard errors
    eta = X @ beta
    mu = logistic(eta)
    mu = np.clip(mu, 1e-8, 1 - 1e-8)
    W = n_trials * mu * (1.0 - mu)
    WX = X * W[:, None]
    XtWX = X.T @ WX

    return beta, XtWX


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in working directory.")

    # Load data using numpy only (no pandas dependency)
    data = np.genfromtxt(
        data_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        missing_values="",
        filling_values=np.nan,
    )

    tooth_class = data["feature1"]
    missing = data["feature3"].astype(float)
    sockets = data["feature4"].astype(float)
    age = data["feature5"].astype(float)
    sex_estimate = data["feature7"].astype(float)
    genus = data["feature8"]

    # Basic cleaning: remove rows with invalid or missing values
    valid_mask = (
        ~np.isnan(missing)
        & ~np.isnan(sockets)
        & ~np.isnan(age)
        & ~np.isnan(sex_estimate)
        & (sockets > 0)
        & (genus != "")
        & (tooth_class != "")
    )

    missing = missing[valid_mask]
    sockets = sockets[valid_mask]
    age = age[valid_mask]
    sex_estimate = sex_estimate[valid_mask]
    genus = genus[valid_mask]
    tooth_class = tooth_class[valid_mask]

    # Define categorical encodings
    genus_levels = sorted(np.unique(genus))
    if "Homo sapiens" in genus_levels:
        reference_genus = "Homo sapiens"
    else:
        reference_genus = genus_levels[0]

    other_genera = [g for g in genus_levels if g != reference_genus]

    n_obs = missing.shape[0]
    X_cols = []
    feature_names: list[str] = []

    # Intercept
    X_cols.append(np.ones(n_obs))
    feature_names.append("intercept")

    # Age
    X_cols.append(age)
    feature_names.append("age")

    # Sex estimate (continuous between 0 and 1)
    X_cols.append(sex_estimate)
    feature_names.append("sex_estimate")

    # Genus dummy variables (reference: Homo sapiens)
    for g in other_genera:
        X_cols.append((genus == g).astype(float))
        feature_names.append(f"genus_{g}")

    # Tooth class dummy variables (reference: Anterior)
    X_cols.append((tooth_class == "Posterior").astype(float))
    feature_names.append("tooth_posterior")
    X_cols.append((tooth_class == "Premolar").astype(float))
    feature_names.append("tooth_premolar")

    X = np.column_stack(X_cols)

    # Fit binomial logistic regression using IRLS
    beta_hat, XtWX = fit_binomial_logistic(X, missing, sockets)

    # Standard errors and p-values for coefficients
    try:
        cov_beta = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov_beta = np.full_like(XtWX, np.nan)

    se_beta = np.sqrt(np.diag(cov_beta))

    genus_effects: dict[str, dict[str, float]] = {}
    alpha = 0.05

    for g in other_genera:
        coef_name = f"genus_{g}"
        idx = feature_names.index(coef_name)
        coef = float(beta_hat[idx])
        se = float(se_beta[idx]) if not np.isnan(se_beta[idx]) else float("nan")
        if np.isnan(se) or se == 0.0:
            pval = float("nan")
        else:
            z = coef / se
            pval = 2.0 * (1.0 - normal_cdf(abs(z)))
        genus_effects[g] = {"coef": coef, "pvalue": pval}

    # Interpret results: whether Homo sapiens has higher AMTL frequency
    # than each non-human genus after adjusting for age, sex, and tooth class.
    evidence_scores = []
    for g, eff in genus_effects.items():
        coef = eff["coef"]
        pval = eff["pvalue"]
        if math.isnan(coef) or math.isnan(pval):
            continue

        # Coefficients are for non-human genera relative to Homo sapiens.
        # Negative coef: that genus has lower AMTL than humans (humans higher).
        if pval < alpha and coef < 0:
            evidence_scores.append(1.0)
        elif pval < alpha and coef > 0:
            evidence_scores.append(-1.0)
        else:
            evidence_scores.append(0.0)

    # Aggregate evidence into a 0-100 Likert-style response.
    if not evidence_scores:
        response_value = 50
        qualitative_conclusion = (
            "Inconclusive: the logistic regression did not yield clear or stable "
            "genus effects, so there is no strong evidence that modern humans differ "
            "in AMTL frequency from non-human primates after accounting for age, sex, "
            "and tooth class."
        )
    else:
        mean_score = float(np.mean(evidence_scores))

        if mean_score > 0:
            # Evidence that humans tend to have higher AMTL
            response_value = int(round(60 + 35 * mean_score))
            qualitative_conclusion = (
                "Yes: the fitted binomial logistic regression indicates that modern humans "
                "have higher frequencies of antemortem tooth loss (AMTL) than the non-human "
                "primate genera considered, after adjusting for age at death, estimated sex, "
                "and tooth class. This is reflected in negative and statistically significant "
                "genus coefficients (relative to Homo sapiens) for one or more non-human genera."
            )
        elif mean_score < 0:
            # Evidence that humans tend to have lower AMTL
            response_value = int(round(40 + 35 * mean_score))
            qualitative_conclusion = (
                "No: the fitted binomial logistic regression indicates that modern humans do "
                "not have higher frequencies of antemortem tooth loss (AMTL) than non-human "
                "primate genera after adjusting for age at death, estimated sex, and tooth class; "
                "some non-human genera instead show higher AMTL frequencies relative to humans."
            )
        else:
            response_value = 50
            qualitative_conclusion = (
                "Uncertain: the fitted binomial logistic regression does not provide strong or "
                "consistent evidence that modern humans differ in AMTL frequency from non-human "
                "primates after accounting for age, sex, and tooth class."
            )

    response_value = int(max(0, min(100, response_value)))

    # Build explanation summarizing data, model, and key genus effects
    explanation_parts = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) compared to non-human primate genera (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?",
        "Data and model: Using the AMTL dataset (1,450 rows in the original file), I modeled the "
        "number of missing teeth out of observable sockets for each specimen and tooth class using "
        "a binomial logistic regression with an intercept, genus indicators (with Homo sapiens as "
        "the reference), age at death, estimated sex, and tooth-class indicators (posterior and "
        "premolar versus anterior).",
    ]

    if genus_effects and reference_genus is not None:
        genus_summaries = []
        for g, eff in genus_effects.items():
            coef = eff["coef"]
            pval = eff["pvalue"]
            if math.isnan(coef) or math.isnan(pval):
                continue
            direction = "lower" if coef < 0 else "higher"
            signif = "statistically significant" if pval < alpha else "not statistically significant"
            genus_summaries.append(
                f"{g} shows {direction} log-odds of AMTL than {reference_genus} "
                f"({signif}, p ≈ {pval:.3f})."
            )
        if genus_summaries:
            explanation_parts.append("Genus effects: " + " ".join(genus_summaries))

    explanation_parts.append(qualitative_conclusion)
    explanation = " ".join(explanation_parts)

    conclusion = {"response": response_value, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
