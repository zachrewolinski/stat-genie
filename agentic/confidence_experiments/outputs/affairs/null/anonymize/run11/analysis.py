import json
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def compute_likert_from_effect(coef: float, pval: float) -> int:
    """
    Map effect direction and significance to a 0-100 Likert score answering:
    'Does having children decrease engagement in extramarital affairs?'
    0 = strong 'No', 100 = strong 'Yes'.
    """
    # Strong evidence
    if pval < 0.01:
        if coef < 0:
            score = 90
        else:
            score = 5
    # Conventional significance
    elif pval < 0.05:
        if coef < 0:
            score = 75
        else:
            score = 15
    # Marginal evidence
    elif pval < 0.10:
        if coef < 0:
            score = 65
        else:
            score = 25
    # No statistically significant evidence: answer should lean toward "No"
    else:
        if coef < 0:
            score = 45
        elif coef > 0:
            score = 35
        else:
            score = 40

    return int(score)


def describe_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    group_stats = df.groupby("feature6")["feature2"].agg(["mean", "median", "std", "count"])
    prevalence = df.groupby("feature6")["any_affair"].mean()
    return group_stats, prevalence


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic descriptive statistics by children status
    group_stats, prevalence = describe_groups(df)

    # Unadjusted logistic regression of any affair on children
    logit_unadj = smf.logit("any_affair ~ C(feature6)", data=df).fit(disp=False)
    coef_unadj = float(logit_unadj.params["C(feature6)[T.yes]"])
    pval_unadj = float(logit_unadj.pvalues["C(feature6)[T.yes]"])
    or_unadj = float(np.exp(coef_unadj))

    # Adjusted logistic regression controlling for key covariates
    # Gender (feature3) is categorical; others are numeric covariates
    formula_adj = (
        "any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    coef_adj = float(logit_adj.params["C(feature6)[T.yes]"])
    pval_adj = float(logit_adj.pvalues["C(feature6)[T.yes]"])
    or_adj = float(np.exp(coef_adj))

    # Likert response is based primarily on the adjusted effect
    response = compute_likert_from_effect(coef_adj, pval_adj)

    # Pull descriptive numbers for explanation
    mean_yes = float(group_stats.loc["yes", "mean"])
    mean_no = float(group_stats.loc["no", "mean"])
    n_yes = int(group_stats.loc["yes", "count"])
    n_no = int(group_stats.loc["no", "count"])
    prev_yes = float(prevalence["yes"] * 100.0)
    prev_no = float(prevalence["no"] * 100.0)

    n_total = int(len(df))

    if pval_adj < 0.01:
        evidence_strength = "strong"
    elif pval_adj < 0.05:
        evidence_strength = "moderate"
    elif pval_adj < 0.10:
        evidence_strength = "weak"
    else:
        evidence_strength = "little"

    if pval_adj < 0.05:
        # Statistically significant results
        if coef_adj < 0:
            direction_text = "associated with *lower* odds of having had an affair"
            answer_text = (
                "there is statistically significant evidence that having children is associated "
                "with a meaningful decrease in engagement in extramarital affairs"
            )
        elif coef_adj > 0:
            direction_text = "associated with *higher* odds of having had an affair"
            answer_text = (
                "there is statistically significant evidence against the hypothesis that "
                "having children decreases engagement; instead, children are associated with "
                "higher odds of affairs"
            )
        else:
            direction_text = (
                "not measurably associated with the odds of having had an affair"
            )
            answer_text = (
                "the data do not show a clear association between having children and "
                "engagement in extramarital affairs"
            )
    else:
        # Not statistically significant: emphasize lack of evidence
        if coef_adj < 0:
            direction_text = (
                "associated with slightly lower but highly uncertain odds of having had an affair"
            )
            answer_text = (
                "the data provide little statistical evidence that having children truly "
                "reduces engagement in extramarital affairs; the estimated decrease is small "
                "and far from statistically significant"
            )
        elif coef_adj > 0:
            direction_text = (
                "associated with slightly higher but highly uncertain odds of having had an affair"
            )
            answer_text = (
                "the data provide little statistical evidence that having children materially "
                "changes engagement in extramarital affairs; the point estimate is slightly "
                "positive but very imprecise"
            )
        else:
            direction_text = (
                "not measurably associated with the odds of having had an affair"
            )
            answer_text = (
                "the data do not provide clear evidence that having children affects "
                "engagement in extramarital affairs in either direction"
            )

    explanation = (
        f"Using the Fair affairs dataset (n={n_total}), I examined whether having children "
        f"decreases engagement in extramarital affairs. Engagement was coded as a binary "
        f"indicator of any extramarital intercourse in the past year (feature2 > 0), and "
        f"children were coded from feature6 ('yes'/'no').\n\n"
        f"Descriptively, the mean affair score was {mean_yes:.3f} for respondents with "
        f"children (n={n_yes}) and {mean_no:.3f} for respondents without children (n={n_no}). "
        f"The prevalence of any affair was {prev_yes:.1f}% among those with children versus "
        f"{prev_no:.1f}% among those without children.\n\n"
        f"I then fit logistic regression models with any affair as the outcome. In an "
        f"unadjusted model with only children status, the coefficient for having children "
        f"was {coef_unadj:.3f}, corresponding to an odds ratio of {or_unadj:.2f} and a "
        f"p-value of {pval_unadj:.3g}. To account for potential confounders, I fit an "
        f"adjusted model controlling for gender, age, years married, religiosity, education, "
        f"occupation, and self-rated marital happiness. In this adjusted model, the "
        f"coefficient for having children was {coef_adj:.3f}, giving an odds ratio of "
        f"{or_adj:.2f} with a p-value of {pval_adj:.3g}.\n\n"
        f"These results indicate {evidence_strength} statistical evidence that having "
        f"children is {direction_text}. In other words, {answer_text}. Based on this "
        f"combination of effect size and statistical significance, I map my answer to a "
        f"value of {response} on a 0–100 Likert scale, where 0 is a strong 'No' and 100 is a "
        f"strong 'Yes' to the question 'Does having children decrease engagement in "
        f"extramarital affairs?'."
    )

    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
