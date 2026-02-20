import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def summarize_by_children(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        mean_has_affair=("has_affair", "mean"),
        count=("has_affair", "size"),
    )
    return grouped


def fit_logit(
    formula: str, df: pd.DataFrame
) -> Tuple[float, float, List[str]]:
    try:
        model = smf.logit(formula, data=df)
        result = model.fit(disp=False)
        # children is coded as yes/no; we care about the effect of having children (yes)
        # relative to the baseline of no children.
        term_name = "C(children)[T.yes]"
        coef = float(result.params.get(term_name, np.nan))
        pval = float(result.pvalues.get(term_name, np.nan))
        return coef, pval, result.summary().as_text().splitlines()
    except Exception:
        return float("nan"), float("nan"), []


def build_explanation(
    group_stats: pd.DataFrame,
    coef_simple: float,
    pval_simple: float,
    coef_full: float,
    pval_full: float,
) -> str:
    parts: List[str] = []

    parts.append(
        "Using 601 married respondents from the Psychology Today survey, "
        "I examined whether having children is associated with engagement "
        "in extramarital affairs."
    )

    if "yes" in group_stats.index and "no" in group_stats.index:
        mean_prob_children = group_stats.loc["yes", "mean_has_affair"]
        mean_prob_nochild = group_stats.loc["no", "mean_has_affair"]
        mean_affairs_children = group_stats.loc["yes", "mean_affairs"]
        mean_affairs_nochild = group_stats.loc["no", "mean_affairs"]

        parts.append(
            "In the raw data, the probability of reporting at least one "
            f"affair is {mean_prob_children:.3f} for respondents with children "
            f"and {mean_prob_nochild:.3f} for those without children, while the "
            f"mean number of affairs is {mean_affairs_children:.2f} versus "
            f"{mean_affairs_nochild:.2f}, respectively."
        )
    else:
        parts.append(
            "The sample does not contain both couples with and without "
            "children, so raw comparisons by children status are limited."
        )

    parts.append(
        "I then modeled the probability of having any affair (affairs > 0) "
        "using logistic regression."
    )

    if np.isfinite(coef_simple):
        parts.append(
            "A logistic regression with only a children indicator gives a "
            f"coefficient of {coef_simple:.3f} (p = {pval_simple:.3f}) for "
            "having children (relative to no children)."
        )
    else:
        parts.append(
            "A simple logistic regression with only a children indicator "
            "could not be reliably estimated."
        )

    if np.isfinite(coef_full):
        parts.append(
            "A logistic regression controlling for age, years married, gender, "
            "religiousness, education, occupation, and self-rated marital "
            f"happiness yields a coefficient of {coef_full:.3f} "
            f"(p = {pval_full:.3f}) for having children."
        )
    else:
        parts.append(
            "A multivariable logistic regression including children and "
            "other covariates could not be reliably estimated."
        )

    return " ".join(parts)


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    group_stats = summarize_by_children(df)

    # Logistic regression with only children.
    coef_simple, pval_simple, _ = fit_logit("has_affair ~ C(children)", df)

    # Logistic regression with controls.
    formula_full = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating"
    )
    coef_full, pval_full, _ = fit_logit(formula_full, df)

    # Evaluate whether having children decreases engagement in affairs.
    mean_prob_children = (
        group_stats.loc["yes", "mean_has_affair"]
        if "yes" in group_stats.index
        else np.nan
    )
    mean_prob_nochild = (
        group_stats.loc["no", "mean_has_affair"]
        if "no" in group_stats.index
        else np.nan
    )

    evidence_signs_negative = 0
    evidence_signs_positive = 0
    evidence_pvals: List[float] = []

    for coef, pval in ((coef_simple, pval_simple), (coef_full, pval_full)):
        if np.isfinite(coef):
            if coef < 0:
                evidence_signs_negative += 1
            elif coef > 0:
                evidence_signs_positive += 1
        if np.isfinite(pval):
            evidence_pvals.append(pval)

    decreases = False
    if (
        evidence_signs_negative > evidence_signs_positive
        and np.isfinite(mean_prob_children)
        and np.isfinite(mean_prob_nochild)
        and mean_prob_children < mean_prob_nochild
    ):
        decreases = True

    response = "Yes" if decreases else "No"

    # Confidence score based on consistency and statistical strength.
    min_p = min(evidence_pvals) if evidence_pvals else float("nan")
    if decreases:
        base_conf = 60.0
        if np.isfinite(min_p):
            if min_p < 0.01:
                base_conf = 90.0
            elif min_p < 0.05:
                base_conf = 80.0
            elif min_p < 0.1:
                base_conf = 70.0
    else:
        base_conf = 70.0
        if np.isfinite(min_p):
            if evidence_signs_positive > 0 and min_p < 0.05:
                base_conf = 90.0
            elif min_p < 0.05:
                base_conf = 80.0

    confidence = int(round(max(0.0, min(100.0, base_conf))))

    explanation_core = build_explanation(
        group_stats, coef_simple, pval_simple, coef_full, pval_full
    )

    if decreases:
        interpretation = (
            "Taken together, the descriptive comparisons and regression results "
            "indicate that respondents with children are meaningfully less "
            "likely to report extramarital affairs than those without "
            "children, so in this sample having children appears to decrease "
            "engagement in affairs."
        )
    else:
        interpretation = (
            "Across both descriptive statistics and regression models, having "
            "children is not associated with a lower likelihood of affairs; "
            "if anything, the estimated effects are small and/or point toward "
            "similar or higher engagement among parents, and statistically "
            "significant negative effects are not supported. Therefore, this "
            "dataset does not provide evidence that having children decreases "
            "engagement in extramarital affairs."
        )

    explanation = f"{explanation_core} {interpretation}"

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

