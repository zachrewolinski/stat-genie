import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_likert_score_from_logit(coef: float, pval: float) -> int:
    """
    Map the children coefficient and p-value from a logistic model
    to a 0–100 Likert score where higher means stronger evidence that
    having children decreases engagement in extramarital affairs.
    """
    # Direction: negative coefficient -> children associated with fewer affairs
    if abs(coef) < 1e-8:
        sign = 0.0
    else:
        sign = np.sign(coef)

    # Evidence component from p-value (p <= 0.5 contributes; larger p -> weaker)
    p_clipped = max(min(pval, 1.0), 0.0)
    evidence = 1.0 - min(p_clipped / 0.5, 1.0)

    # Effect size component from absolute log-odds (saturate beyond 1.5)
    effect_magnitude = min(abs(coef), 1.5) / 1.5

    strength = 0.5 * evidence + 0.5 * effect_magnitude

    base = 50.0
    score = base + sign * strength * 50.0
    score = max(0.0, min(100.0, score))
    return int(round(score))


def main() -> None:
    # Load data
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define outcome: any extramarital sexual intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Children indicator: 1 if there are children in the marriage, 0 otherwise
    df["children"] = df["feature6"].astype(str).str.lower().eq("yes").astype(int)

    # Drop rows with missing values in variables of interest (defensive, though data is clean)
    df_model = df[["any_affair", "children"]].dropna()

    # Descriptive statistics by children status
    group_any = df_model.groupby("children")["any_affair"].agg(["mean", "count"])
    group_freq = df.groupby("children")["feature2"].mean()

    # Logistic regression: any_affair ~ children
    y = df_model["any_affair"]
    X = sm.add_constant(df_model["children"])
    logit_model = sm.Logit(y, X).fit(disp=False)

    coef_children = float(logit_model.params["children"])
    pval_children = float(logit_model.pvalues["children"])
    odds_ratio = float(np.exp(coef_children))

    # Map to Likert score
    score = compute_likert_score_from_logit(coef_children, pval_children)

    # Prepare explanation text
    n_total = int(len(df_model))
    n_children1 = int(group_any.loc[1, "count"])
    n_children0 = int(group_any.loc[0, "count"])
    prop_children1 = float(group_any.loc[1, "mean"])
    prop_children0 = float(group_any.loc[0, "mean"])
    mean_freq_children1 = float(group_freq.loc[1])
    mean_freq_children0 = float(group_freq.loc[0])

    direction_text = (
        "a lower likelihood of any extramarital intercourse when there are children in the marriage"
        if coef_children < 0
        else "a higher likelihood of any extramarital intercourse when there are children in the marriage"
        if coef_children > 0
        else "no clear difference in likelihood of extramarital intercourse between marriages with and without children"
    )

    explanation = (
        "I analyzed a dataset of married individuals from the Fair (1978) extramarital affairs study, "
        "containing 601 observations with variables on frequency of extramarital sexual intercourse in the past year "
        "(feature2) and whether there are children in the marriage (feature6, yes/no), along with demographic and "
        "marital covariates. I defined engagement in extramarital affairs as a binary outcome indicating whether the "
        "respondent reported any extramarital sexual intercourse in the past year (feature2 > 0) and constructed a "
        "children indicator equal to 1 when feature6 == 'yes'. "
        f"In the data used for modelling (n = {n_total}), there were {n_children1} individuals in marriages with "
        f"children and {n_children0} in marriages without children. The proportion reporting any extramarital "
        f"intercourse was {prop_children1:.1%} among those with children versus {prop_children0:.1%} among those "
        f"without children, and the mean coded frequency of intercourse (feature2) was {mean_freq_children1:.2f} "
        f"with children versus {mean_freq_children0:.2f} without children. "
        "To formally assess the association between children and engagement in extramarital affairs, I fit a logistic "
        "regression model with the binary outcome (any affair vs. none) and a single predictor for the presence of "
        "children. The estimated log-odds coefficient for the children indicator was "
        f"{coef_children:.3f}, corresponding to an odds ratio of {odds_ratio:.2f}, with p-value {pval_children:.4f}. "
        f"This indicates {direction_text}. "
        "I then mapped the sign and statistical strength of this association to a 0–100 Likert scale, where higher "
        "values represent stronger evidence that having children decreases engagement in extramarital affairs and "
        "lower values represent evidence that children do not decrease (or may even increase) such engagement. "
        f"The resulting score of {score} reflects the combined direction and strength of the estimated effect: values "
        "above 50 indicate evidence that having children is associated with fewer extramarital affairs, values near "
        "50 indicate little or ambiguous evidence, and values below 50 indicate evidence in the opposite direction. "
        "This score, together with the descriptive statistics and logistic regression results, forms the basis for my "
        "answer to the research question of whether having children decreases engagement in extramarital affairs."
    )

    result = {
        "response": int(score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

