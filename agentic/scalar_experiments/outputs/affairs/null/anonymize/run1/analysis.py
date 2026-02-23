import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital affair in the past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in the marriage (1 = yes, 0 = no)
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    # Basic descriptive comparison
    prop_affair_children = (
        df.loc[df["children_yes"] == 1, "has_affair"].mean()
    )
    prop_affair_no_children = (
        df.loc[df["children_yes"] == 0, "has_affair"].mean()
    )

    # 2x2 chi-square test of independence
    contingency = pd.crosstab(df["children_yes"], df["has_affair"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

    # Unadjusted logistic regression: has_affair ~ children_yes
    model_unadj = smf.logit("has_affair ~ children_yes", data=df).fit(disp=False)
    coef_unadj = model_unadj.params["children_yes"]
    p_unadj = model_unadj.pvalues["children_yes"]
    or_unadj = float(np.exp(coef_unadj))

    # Adjusted logistic regression controlling for key covariates
    # Gender (categorical), age, years married, religiosity, education,
    # occupation, and self-rated marriage quality.
    model_adj = smf.logit(
        "has_affair ~ children_yes + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10",
        data=df,
    ).fit(disp=False)
    coef_adj = model_adj.params["children_yes"]
    p_adj = model_adj.pvalues["children_yes"]
    or_adj = float(np.exp(coef_adj))

    # Determine Likert-scale response (0–100, 0 = strong No, 100 = strong Yes)
    response = score_evidence(coef_adj, p_adj, or_adj)

    explanation = build_explanation(
        prop_affair_children=prop_affair_children,
        prop_affair_no_children=prop_affair_no_children,
        chi2=chi2,
        p_chi2=p_chi2,
        coef_unadj=coef_unadj,
        p_unadj=p_unadj,
        or_unadj=or_unadj,
        coef_adj=coef_adj,
        p_adj=p_adj,
        or_adj=or_adj,
        response=response,
    )

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


def score_evidence(coef_adj: float, p_adj: float, or_adj: float) -> int:
    """
    Map statistical evidence to a 0–100 Likert scale.

    Positive answer = children decrease engagement in extramarital affairs,
    i.e., odds ratio < 1 (negative coefficient for children_yes).
    """
    direction_negative = coef_adj < 0

    # Default: uncertain / no clear evidence
    score = 50

    if direction_negative:
        # Evidence that children are associated with *fewer* affairs.
        if p_adj < 0.001:
            if or_adj < 0.6:
                score = 95
            elif or_adj < 0.8:
                score = 88
            else:
                score = 80
        elif p_adj < 0.05:
            if or_adj < 0.6:
                score = 88
            elif or_adj < 0.8:
                score = 80
            else:
                score = 70
        else:
            # Directionally consistent but not statistically significant
            if or_adj < 1.0:
                score = 60
            else:
                score = 50
    else:
        # Evidence that children are associated with *more* affairs
        # (opposite of the hypothesized decrease).
        if p_adj < 0.001:
            if or_adj > 1.6:
                score = 5
            elif or_adj > 1.2:
                score = 12
            else:
                score = 20
        elif p_adj < 0.05:
            if or_adj > 1.6:
                score = 12
            elif or_adj > 1.2:
                score = 20
            else:
                score = 30
        else:
            score = 50

    # Ensure integer within [0, 100]
    score = max(0, min(100, int(round(score))))
    return score


def build_explanation(
    *,
    prop_affair_children: float,
    prop_affair_no_children: float,
    chi2: float,
    p_chi2: float,
    coef_unadj: float,
    p_unadj: float,
    or_unadj: float,
    coef_adj: float,
    p_adj: float,
    or_adj: float,
    response: int,
) -> str:
    direction_adj = "decrease" if coef_adj < 0 else "increase"

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"In this sample of 601 first-marriage individuals, the proportion reporting any "
        f"extramarital intercourse in the past year was "
        f"{prop_affair_no_children:.3f} among those without children and "
        f"{prop_affair_children:.3f} among those with children.\n"
        f"A chi-square test of independence between children-in-marriage and any affair "
        f"yielded χ² = {chi2:.2f} (p = {p_chi2:.4f}).\n"
        f"Unadjusted logistic regression (any affair ~ children) gave a coefficient for having "
        f"children of {coef_unadj:.3f} (odds ratio = {or_unadj:.3f}, p = {p_unadj:.4f}).\n"
        f"Adjusting for gender, age, years married, religiosity, education, occupation, and "
        f"self-rated marital happiness, the coefficient for having children was {coef_adj:.3f} "
        f"(odds ratio = {or_adj:.3f}, p = {p_adj:.4f}), implying that children are associated "
        f"with a relative {direction_adj} in the odds of having an affair.\n"
        f"Combining the direction and statistical significance of the adjusted effect, the "
        f"evidence corresponds to a Likert-scale response of {response} on a 0–100 scale "
        f"(0 = strong 'No', 100 = strong 'Yes') to the question of whether having children "
        f"reduces engagement in extramarital affairs."
    )

    return explanation


if __name__ == "__main__":
    main()

