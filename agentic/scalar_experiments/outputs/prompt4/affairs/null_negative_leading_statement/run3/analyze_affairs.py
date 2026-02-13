import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Binary indicator for any extramarital affair in the past year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries by children status.
    grp = df.groupby("children", observed=True)
    mean_affair_rate = grp["has_affair"].mean()
    mean_affair_count = grp["affairs"].mean()

    # Ensure both levels are present for interpretation; if not, analysis is limited.
    children_levels = sorted(mean_affair_rate.index.tolist())

    # Logistic regression: any affair ~ children + key controls.
    # This mirrors common analyses of the Fair affairs dataset.
    model = smf.logit(
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)

    coef_name = "C(children)[T.yes]"
    coef_children = float(model.params.get(coef_name, np.nan))
    pval_children = float(model.pvalues.get(coef_name, np.nan))
    odds_ratio = float(np.exp(coef_children)) if np.isfinite(coef_children) else float("nan")

    # Difference in raw affair rates (children=yes minus children=no) where available.
    diff_rate = np.nan
    rate_no = np.nan
    rate_yes = np.nan
    mean_no = np.nan
    mean_yes = np.nan

    if {"no", "yes"} <= set(children_levels):
        rate_no = float(mean_affair_rate["no"])
        rate_yes = float(mean_affair_rate["yes"])
        mean_no = float(mean_affair_count["no"])
        mean_yes = float(mean_affair_count["yes"])
        diff_rate = rate_yes - rate_no

    # Map statistical evidence to a 0–100 scale where
    # 0 = strong "No" (children do NOT decrease affairs)
    # 100 = strong "Yes" (children DO decrease affairs).
    score = map_evidence_to_scale(coef_children, pval_children, diff_rate)

    explanation = build_explanation(
        rate_no=rate_no,
        rate_yes=rate_yes,
        mean_no=mean_no,
        mean_yes=mean_yes,
        coef_children=coef_children,
        pval_children=pval_children,
        odds_ratio=odds_ratio,
        diff_rate=diff_rate,
        score=score,
    )

    conclusion = {"response": int(score), "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def map_evidence_to_scale(coef_children: float, pval_children: float, diff_rate: float) -> int:
    """
    Convert the statistical evidence into a 0–100 Likert score.

    Higher scores mean stronger evidence that having children
    decreases engagement in extramarital affairs.
    """
    if not np.isfinite(coef_children) or not np.isfinite(pval_children):
        return 50

    # Negative coefficient and lower raw affair rate with children
    # both point toward a protective effect (answer "Yes").
    if coef_children < 0 and (not np.isnan(diff_rate) and diff_rate < 0):
        if pval_children < 0.001:
            return 95
        if pval_children < 0.01:
            return 90
        if pval_children < 0.05:
            return 80
        return 65

    # Positive coefficient and higher raw affair rate with children
    # both point toward increased engagement (answer "No").
    if coef_children > 0 and (not np.isnan(diff_rate) and diff_rate > 0):
        if pval_children < 0.001:
            return 5
        if pval_children < 0.01:
            return 10
        if pval_children < 0.05:
            return 20
        return 35

    # Mixed or weak evidence.
    return 50


def build_explanation(
    rate_no: float,
    rate_yes: float,
    mean_no: float,
    mean_yes: float,
    coef_children: float,
    pval_children: float,
    odds_ratio: float,
    diff_rate: float,
    score: int,
) -> str:
    parts = []

    # Restate the question and scale.
    parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs? "
        "The response is given on a 0–100 scale, where 0 means a strong 'No' and 100 means a strong 'Yes'."
    )

    # Descriptive statistics.
    if np.isfinite(rate_no) and np.isfinite(rate_yes):
        parts.append(
            f"Descriptively, the proportion of respondents reporting at least one affair in the past year "
            f"is {rate_no:.1%} for couples without children and {rate_yes:.1%} for couples with children."
        )
    if np.isfinite(mean_no) and np.isfinite(mean_yes):
        parts.append(
            f"The average number of affairs is {mean_no:.2f} without children versus {mean_yes:.2f} with children."
        )

    # Regression-based inference.
    if np.isfinite(coef_children) and np.isfinite(pval_children) and np.isfinite(odds_ratio):
        direction = "decrease" if coef_children < 0 else "increase"
        parts.append(
            "To adjust for other factors, I fit a logistic regression model predicting whether a respondent had any "
            "extramarital affair from children status, gender, age, years married, religiousness, education, "
            "occupation, and self-rated marital happiness."
        )
        parts.append(
            f"In this model, the coefficient for having children (relative to not having children) is {coef_children:.3f} "
            f"with an odds ratio of {odds_ratio:.3f} and p-value {pval_children:.3g}, indicating that, "
            f"after controlling for these covariates, having children is associated with a relative {direction} in "
            "the odds of reporting an extramarital affair."
        )

    # Overall conclusion.
    if score >= 70:
        qualitative = (
            "overall, the data provide meaningful evidence that having children is associated with lower engagement "
            "in extramarital affairs."
        )
        answer = "Yes"
    elif score <= 30:
        qualitative = (
            "overall, the data suggest that having children does not decrease extramarital affairs and may even be "
            "associated with similar or higher engagement."
        )
        answer = "No"
    else:
        qualitative = (
            "overall, the evidence is mixed or weak, so the data do not clearly show that having children decreases "
            "extramarital affairs."
        )
        answer = "Uncertain"

    parts.append(
        f"Balancing the descriptive differences and the regression results, I assign a score of {score} on the "
        f"0–100 scale, which corresponds to the answer '{answer}' to the question of whether having children "
        "decreases engagement in extramarital affairs."
    )

    return " ".join(parts)


if __name__ == "__main__":
    main()

