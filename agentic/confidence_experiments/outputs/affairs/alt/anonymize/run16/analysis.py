import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome: any extramarital affair in past year (0 = none, 1 = at least one)
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Predictor of interest: children in the marriage (1 = yes, 0 = no)
    df["children"] = df["feature6"].str.lower().eq("yes").astype(int)

    # Covariates
    df["male"] = df["feature3"].str.lower().eq("male").astype(int)

    df_model = df.dropna(
        subset=[
            "any_affair",
            "children",
            "male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    )

    # 2x2 table: children (rows) by any_affair (cols)
    ct = pd.crosstab(df_model["children"], df_model["any_affair"])
    ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    chi2, p_chi2, dof, expected = chi2_contingency(ct.values)

    n_no_child = ct.loc[0].sum()
    n_child = ct.loc[1].sum()
    rate_no_child = ct.loc[0, 1] / n_no_child if n_no_child > 0 else np.nan
    rate_child = ct.loc[1, 1] / n_child if n_child > 0 else np.nan
    diff_rate = rate_child - rate_no_child

    # Logistic regression models for any_affair
    model_unadj = smf.logit("any_affair ~ children", data=df_model).fit(disp=0)
    model_adj = smf.logit(
        "any_affair ~ children + male + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
        data=df_model,
    ).fit(disp=0)

    coef_unadj = float(model_unadj.params["children"])
    p_unadj = float(model_unadj.pvalues["children"])
    or_unadj = float(np.exp(coef_unadj))

    coef_adj = float(model_adj.params["children"])
    p_adj = float(model_adj.pvalues["children"])
    or_adj = float(np.exp(coef_adj))

    ci_adj = model_adj.conf_int().loc["children"]
    or_ci_low = float(np.exp(ci_adj[0]))
    or_ci_high = float(np.exp(ci_adj[1]))

    # Affair frequency by children status
    mean_freq = df_model.groupby("children")["feature2"].mean()
    mean_no_child = float(mean_freq.get(0, np.nan))
    mean_child = float(mean_freq.get(1, np.nan))
    diff_mean_freq = mean_child - mean_no_child

    response, qualitative = score_relationship(
        p_chi2=p_chi2,
        p_unadj=p_unadj,
        p_adj=p_adj,
        coef_adj=coef_adj,
        or_adj=or_adj,
        diff_rate=diff_rate,
    )

    explanation = build_explanation(
        rate_no_child=rate_no_child,
        rate_child=rate_child,
        diff_rate=diff_rate,
        mean_no_child=mean_no_child,
        mean_child=mean_child,
        diff_mean_freq=diff_mean_freq,
        chi2=chi2,
        p_chi2=p_chi2,
        or_unadj=or_unadj,
        p_unadj=p_unadj,
        or_adj=or_adj,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        p_adj=p_adj,
        qualitative=qualitative,
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


def score_relationship(
    p_chi2: float,
    p_unadj: float,
    p_adj: float,
    coef_adj: float,
    or_adj: float,
    diff_rate: float,
) -> tuple[int, str]:
    """
    Map the statistical evidence to a 0-100 Likert score answering:
    \"Does having children decrease engagement in extramarital affairs?\"
    """
    direction = "decrease" if coef_adj < 0 else "increase"

    pvals = [p_chi2, p_unadj, p_adj]
    num_sig = sum(p < 0.05 for p in pvals)

    if direction == "decrease":
        # Evidence that children reduce affairs
        if num_sig >= 3 and p_adj < 0.01 and or_adj < 0.8:
            score = 85
            qualitative = "strong_yes"
        elif num_sig >= 2 and p_adj < 0.05 and or_adj < 0.9:
            score = 75
            qualitative = "yes"
        elif any(p < 0.05 for p in pvals) and or_adj < 1.0:
            score = 65
            qualitative = "weak_yes"
        elif num_sig == 0 and p_adj >= 0.1:
            score = 40
            qualitative = "no_evidence_for_decrease"
        else:
            score = 55
            qualitative = "very_weak_yes_or_uncertain"
    else:
        # Children associated with same or higher affair rate
        if num_sig >= 2 and p_adj < 0.05 and or_adj > 1.1:
            score = 10
            qualitative = "strong_no_opposite_direction"
        elif any(p < 0.05 for p in pvals) and or_adj > 1.0:
            score = 20
            qualitative = "no_evidence_of_decrease_some_evidence_of_increase"
        elif num_sig == 0 and p_adj >= 0.1:
            score = 30
            qualitative = "no_clear_relationship"
        else:
            score = 35
            qualitative = "lean_no"

    score = int(min(max(round(score), 0), 100))
    return score, qualitative


def build_explanation(
    rate_no_child: float,
    rate_child: float,
    diff_rate: float,
    mean_no_child: float,
    mean_child: float,
    diff_mean_freq: float,
    chi2: float,
    p_chi2: float,
    or_unadj: float,
    p_unadj: float,
    or_adj: float,
    or_ci_low: float,
    or_ci_high: float,
    p_adj: float,
    qualitative: str,
) -> str:
    if qualitative in {
        "strong_yes",
        "yes",
        "weak_yes",
        "very_weak_yes_or_uncertain",
    }:
        headline = (
            "There is some evidence that having children is associated "
            "with a lower likelihood of extramarital affairs."
        )
    else:
        headline = (
            "The data do not support the claim that having children reduces "
            "engagement in extramarital affairs."
        )

    explanation = (
        f"{headline} "
        f"In this sample, the proportion reporting any affair in the past year was "
        f"{rate_no_child:.3f} for couples without children and {rate_child:.3f} for couples with children "
        f"(difference = {diff_rate:+.3f}). "
        f"The average affair frequency (on the 0–12+ scale) was {mean_no_child:.2f} without children "
        f"versus {mean_child:.2f} with children (difference = {diff_mean_freq:+.2f}). "
        f"A chi-squared test of the 2×2 children-by-any-affair table yielded χ² = {chi2:.2f} with p = {p_chi2:.3f}. "
        f"A logistic regression of any affair on children alone gave an odds ratio of {or_unadj:.2f} for couples with children "
        f"(p = {p_unadj:.3f}). "
        f"In a multivariable logistic model adjusting for gender, age, years married, religiousness, education, occupation, "
        f"and marital satisfaction, the odds ratio for couples with children was {or_adj:.2f} with a 95% confidence interval "
        f"of [{or_ci_low:.2f}, {or_ci_high:.2f}] (p = {p_adj:.3f}). "
    )

    if qualitative in {"strong_yes", "yes"}:
        explanation += (
            "Across these analyses, the effect of having children is consistently negative and statistically significant, "
            "indicating that, holding other factors constant, parents in this sample are meaningfully less likely to report extramarital affairs."
        )
    elif qualitative in {"weak_yes", "very_weak_yes_or_uncertain"}:
        explanation += (
            "These results point toward a modest negative association between having children and affairs, "
            "but the effect size is small and/or only marginally significant, so the evidence for a protective effect of children is limited."
        )
    elif qualitative == "strong_no_opposite_direction":
        explanation += (
            "Here, having children is actually associated with a significantly higher likelihood of affairs, "
            "so the data support the opposite of the proposed decrease."
        )
    elif qualitative == "no_evidence_of_decrease_some_evidence_of_increase":
        explanation += (
            "Some models suggest a small increase in affair likelihood among couples with children, "
            "and none of the analyses support a statistically reliable decrease, so the data do not support the hypothesis."
        )
    elif qualitative in {"no_clear_relationship", "lean_no"}:
        explanation += (
            "Effect estimates are close to null and not consistently statistically significant, "
            "so the sample does not provide clear evidence that having children changes the likelihood of extramarital affairs in either direction."
        )

    return explanation


if __name__ == "__main__":
    main()

