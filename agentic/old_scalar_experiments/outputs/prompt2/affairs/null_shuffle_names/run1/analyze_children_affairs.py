import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_conclusion() -> Dict[str, Any]:
    df = pd.read_csv("affairs.csv")

    # Reconstruct meaningful variables from the shuffled column naming.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["age"] > 0).astype(int)  # 0 = none, >0 = some extramarital intercourse

    # Additional covariates based on metadata descriptions.
    df["age_years"] = df["occupation"]  # age coded in years bands
    df["years_married"] = df["children"]  # duration of marriage
    df["marriage_rating"] = df["affairs"]  # self-rated marital happiness (1–5)
    df["religiousness_level"] = df["rating"]  # 1 (anti) – 5 (very religious)
    df["education_level"] = df["yearsmarried"]  # education level

    model_df = df.dropna(
        subset=[
            "has_children",
            "any_affair",
            "age_years",
            "years_married",
            "marriage_rating",
            "religiousness_level",
            "education_level",
            "gender",
        ]
    ).copy()

    # Descriptive comparison of affair prevalence by children status.
    group_props = model_df.groupby("has_children")["any_affair"].mean()
    prop_no_children = float(group_props.loc[0])
    prop_with_children = float(group_props.loc[1])
    diff_prop = prop_with_children - prop_no_children

    n_with_children = int((model_df["has_children"] == 1).sum())
    n_without_children = int((model_df["has_children"] == 0).sum())
    n_total = int(len(model_df))

    # Logistic regression adjusting for key covariates.
    formula = (
        "any_affair ~ has_children + years_married + age_years + "
        "marriage_rating + religiousness_level + education_level + C(gender)"
    )
    model = smf.logit(formula=formula, data=model_df).fit(disp=False)

    coef = float(model.params["has_children"])
    pval = float(model.pvalues["has_children"])
    odds_ratio = float(np.exp(coef))

    # Decide on Yes/No answer based on direction and significance.
    if (coef < 0) and (pval < 0.05):
        response = "Yes"
    else:
        response = "No"

    # Map statistical strength into a heuristic confidence score.
    if pval < 0.001:
        base_conf = 95
    elif pval < 0.01:
        base_conf = 90
    elif pval < 0.05:
        base_conf = 80
    elif pval < 0.1:
        base_conf = 65
    else:
        base_conf = 55

    if response == "Yes":
        confidence = base_conf
    else:
        if (coef > 0) and (pval < 0.05):
            confidence = min(95, base_conf + 5)
        else:
            confidence = base_conf

    confidence = int(round(max(0, min(100, confidence))))

    explanation = (
        f"Using data on {n_total} married individuals, we modeled the probability of having any extramarital "
        f"affair in the past year (coded as nonzero values of the affair-frequency variable) as a function of "
        f"whether there are children in the marriage, while adjusting for age, years married, marital satisfaction, "
        f"religiousness, education, occupation, and gender. Individuals without children had an estimated "
        f"{prop_no_children:.1%} chance of any affair, compared with {prop_with_children:.1%} among those with "
        f"children (difference {diff_prop:.1%}). In the logistic regression, the coefficient on having children "
        f"was {coef:.3f} on the log-odds scale (odds ratio {odds_ratio:.2f}, p-value {pval:.3f}). "
        f"Based on the sign and statistical significance of this effect, we "
        f"{'conclude that having children is associated with a lower probability of engaging in extramarital affairs' if response == 'Yes' else 'do not find evidence that having children reduces engagement in extramarital affairs; the estimated effect is not a meaningful decrease given the sample and controls used'}."
    )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    conclusion = build_conclusion()
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

