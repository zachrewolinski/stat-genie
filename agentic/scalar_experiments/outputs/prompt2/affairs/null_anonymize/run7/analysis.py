import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # feature2: frequency of extramarital intercourse in past year
    # feature6: children in marriage ("yes"/"no")
    df = df.copy()
    df["has_affair"] = (df["feature2"] > 0).astype(int)
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    # Basic descriptive stats
    grp = df.groupby("children_yes")
    mean_freq = grp["feature2"].mean()
    prop_affair = grp["has_affair"].mean()
    n = grp.size()

    # Difference in means (frequency) and proportions (any affair)
    diff_freq = float(mean_freq.loc[0] - mean_freq.loc[1])  # no children minus children
    diff_prop = float(prop_affair.loc[0] - prop_affair.loc[1])

    # Simple linear probability model for having any affair, controlling for key covariates
    X = df[[
        "children_yes",  # main variable of interest
        "feature3",      # gender (categorical)
        "feature4",      # age
        "feature5",      # years married
        "feature7",      # religiousness
        "feature8",      # education
        "feature9",      # occupation
        "feature10",     # marriage rating
    ]].copy()

    # Encode gender
    X["gender_male"] = (df["feature3"] == "male").astype(int)
    X = X.drop(columns=["feature3"])

    X = sm.add_constant(X)
    y = df["has_affair"]

    model = sm.OLS(y, X).fit()
    children_coef = float(model.params["children_yes"])
    children_pval = float(model.pvalues["children_yes"])

    # Heuristic decision rule:
    # If having children is associated with lower affair engagement (negative differences and coefficient)
    # and effects are reasonably clear (p-value), answer "Yes", else "No".
    decreases = diff_freq < 0 and diff_prop < 0 and children_coef < 0

    # Map evidence strength to confidence
    confidence = 50
    if decreases:
        confidence = 70
        if children_pval < 0.1:
            confidence = 80
        if children_pval < 0.05:
            confidence = 90
        if children_pval < 0.01:
            confidence = 95
    else:
        # Some evidence in opposite direction or ambiguous
        confidence = 60
        if children_pval < 0.1:
            confidence = 70
        if children_pval < 0.05:
            confidence = 80

    response = "Yes" if decreases else "No"

    explanation_parts = []

    explanation_parts.append(
        "We coded affair engagement as a binary indicator (any non-zero "
        "frequency of extramarital intercourse in the past year) and also "
        "looked at the average reported frequency."
    )

    explanation_parts.append(
        f"Among respondents without children (n={int(n.loc[0])}), the mean affair "
        f"frequency was {mean_freq.loc[0]:.3f} and the proportion having any affair "
        f"was {prop_affair.loc[0]:.3f}. Among respondents with children (n={int(n.loc[1])}), "
        f"the mean frequency was {mean_freq.loc[1]:.3f} and the proportion having any affair "
        f"was {prop_affair.loc[1]:.3f}. The differences (no children minus children) were "
        f"{diff_freq:.3f} in mean frequency and {diff_prop:.3f} in the probability of any affair."
    )

    explanation_parts.append(
        "We then estimated a linear probability model for having any affair, "
        "including an indicator for having children along with controls for gender, "
        "age, years married, religiousness, education, occupation, and self-rated "
        f"marital happiness. The coefficient on the children indicator was {children_coef:.3f} "
        f"with p-value {children_pval:.3f}."
    )

    if decreases:
        explanation_parts.append(
            "Because the descriptive differences and the regression coefficient for "
            "having children are consistently negative (indicating lower affair "
            "engagement among those with children), we conclude that having children "
            "is associated with decreased engagement in extramarital affairs in this "
            "sample, though the result is observational and may not be strictly causal."
        )
    else:
        explanation_parts.append(
            "Because either the descriptive differences or the regression coefficient "
            "do not consistently indicate lower affair engagement among those with "
            "children, the evidence does not support the claim that having children "
            "decreases extramarital affairs in this sample."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
