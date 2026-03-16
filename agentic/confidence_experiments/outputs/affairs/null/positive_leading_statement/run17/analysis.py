import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def fit_logit(formula: str, data: pd.DataFrame, param_name: str):
    try:
        model = smf.logit(formula, data=data).fit(disp=False)
    except PerfectSeparationError:
        return np.nan, np.nan, np.nan

    coef = model.params.get(param_name, np.nan)
    pval = model.pvalues.get(param_name, np.nan)
    odds_ratio = float(np.exp(coef)) if pd.notnull(coef) else np.nan
    return float(coef), float(pval), odds_ratio


def compute_score(coef: float, pval: float) -> int:
    base = 50
    if coef is None or pval is None or np.isnan(coef) or np.isnan(pval):
        return base

    if pval < 0.01:
        delta = 30
    elif pval < 0.05:
        delta = 20
    elif pval < 0.1:
        delta = 10
    else:
        delta = 5

    if coef < 0:
        score = base + delta
    else:
        score = base - delta

    score = int(round(max(0, min(100, score))))
    return score


def build_explanation(
    grouped: pd.DataFrame,
    coef_unadj: float,
    pval_unadj: float,
    or_unadj: float,
    coef_adj: float,
    pval_adj: float,
    or_adj: float,
    score: int,
) -> str:
    lines = []

    # Descriptive statistics
    for children_value, row in grouped.iterrows():
        lines.append(
            f"For couples with children = {children_value}, "
            f"the mean affairs score is {row['mean_affairs']:.2f}, "
            f"and {row['prop_any_affair'] * 100:.1f}% reported at least one affair "
            f"(n = {int(row['count'])})."
        )

    # Unadjusted model interpretation
    if not np.isnan(coef_unadj) and not np.isnan(pval_unadj):
        direction_unadj = (
            "lower" if coef_unadj < 0 else "higher"
        )
        lines.append(
            "Using a logistic regression of any affair on the presence of children "
            f"without additional controls, couples with children have {direction_unadj} "
            f"odds of reporting an affair than couples without children "
            f"(odds ratio ≈ {or_unadj:.2f}, p-value ≈ {pval_unadj:.3f})."
        )

    # Adjusted model interpretation
    if not np.isnan(coef_adj) and not np.isnan(pval_adj):
        direction_adj = "lower" if coef_adj < 0 else "higher"
        strength = (
            "strong"
            if pval_adj < 0.01
            else "moderate"
            if pval_adj < 0.05
            else "weak"
            if pval_adj < 0.1
            else "little"
        )
        lines.append(
            "Controlling for gender, age, years married, religiousness, education, "
            "occupation, and marital rating in a logistic regression, couples with "
            f"children have {direction_adj} odds of reporting an affair than couples "
            f"without children (adjusted odds ratio ≈ {or_adj:.2f}, "
            f"p-value ≈ {pval_adj:.3f}), providing {strength} statistical evidence "
            "about the association."
        )

        if coef_adj < 0:
            overall = (
                "Overall, the adjusted model suggests that having children is "
                "associated with a decrease in the likelihood of reporting an "
                "extramarital affair, although the magnitude and certainty of this "
                "effect depend on the exact p-value and confidence interval."
            )
        else:
            overall = (
                "Overall, the adjusted model does not support the claim that having "
                "children decreases the likelihood of reporting an extramarital "
                "affair; if anything, the estimated association is in the opposite "
                "direction or too small and uncertain to draw a firm conclusion."
            )
        lines.append(overall)
    else:
        lines.append(
            "Adjusted logistic regression results could not be reliably estimated, "
            "so the conclusion rests primarily on descriptive comparisons."
        )

    if score > 50:
        qualitative = (
            "Taken together, these findings offer overall support for the claim "
            "that having children is associated with lower engagement in "
            "extramarital affairs, though this support is calibrated by the "
            "strength of statistical evidence."
        )
    elif score < 50:
        qualitative = (
            "Taken together, these findings do not support the claim that having "
            "children decreases engagement in extramarital affairs; if an effect "
            "exists, it is small or uncertain in this sample."
        )
    else:
        qualitative = (
            "Taken together, the evidence is essentially inconclusive about whether "
            "having children decreases engagement in extramarital affairs in this "
            "sample."
        )

    lines.append(
        qualitative
        + f" On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"this analysis corresponds to a score of {score}."
    )

    return " ".join(lines)


def main() -> None:
    df = pd.read_csv("affairs.csv")
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    grouped = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        prop_any_affair=("has_affair", "mean"),
        count=("affairs", "size"),
    )

    coef_unadj, pval_unadj, or_unadj = fit_logit(
        "has_affair ~ C(children)", df, "C(children)[T.yes]"
    )

    coef_adj, pval_adj, or_adj = fit_logit(
        "has_affair ~ C(children) + gender + age + yearsmarried + religiousness "
        "+ education + occupation + rating",
        df,
        "C(children)[T.yes]",
    )

    score = compute_score(coef_adj, pval_adj)

    explanation = build_explanation(
        grouped,
        coef_unadj,
        pval_unadj,
        or_unadj,
        coef_adj,
        pval_adj,
        or_adj,
        score,
    )

    output = {"response": score, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

