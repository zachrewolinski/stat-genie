import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "affairs.csv"

    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital affairs in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure children is coded as binary 1=yes, 0=no
    df = df[df["children"].isin(["yes", "no"])].copy()
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Simple descriptive statistics by children status
    grouped = df.groupby("children_yes")["has_affair"].agg(["mean", "sum", "count"])

    mean_no_children = float(grouped.loc[0, "mean"])
    mean_with_children = float(grouped.loc[1, "mean"])

    # Difference in proportions (no children minus with children)
    diff = mean_no_children - mean_with_children

    # Logistic regression controlling for a few key covariates that exist in this file
    covariates = []
    for col in ["age", "yearsmarried", "religiousness", "rating", "gender"]:
        if col in df.columns:
            covariates.append(col)

    model_effect = None
    p_value = None

    if covariates:
        design = df.copy()
        # Encode gender if present
        if "gender" in design.columns:
            design = pd.get_dummies(design, columns=["gender"], drop_first=True)

        x_cols = ["children_yes"]
        for col in covariates:
            if col != "gender" and col in design.columns:
                x_cols.append(col)
        X = design[x_cols]
        X = sm.add_constant(X, has_constant="add")
        y = design["has_affair"]

        logit_model = sm.Logit(y, X).fit(disp=False)
        if "children_yes" in logit_model.params.index:
            model_effect = float(logit_model.params["children_yes"])
            p_value = float(logit_model.pvalues["children_yes"])

    # Map evidence to a 0–100 Likert response where higher means stronger "Yes,
    # having children decreases engagement in extramarital affairs".
    # Start from direction of descriptive difference.
    response: int
    explanation_parts = []

    explanation_parts.append(
        f"Proportion with any affairs is {mean_no_children:.3f} for couples without children "
        f"and {mean_with_children:.3f} for couples with children."
    )
    explanation_parts.append(
        f"The raw difference in proportions (no children minus children) is {diff:.3f}."
    )

    if model_effect is not None and p_value is not None:
        explanation_parts.append(
            "A logistic regression predicting any affair from children and basic controls "
            f"(age, years married, religiousness, rating, gender) gives a coefficient on "
            f"having children of {model_effect:.3f} with p-value {p_value:.4f}."
        )

    # Determine qualitative conclusion
    if diff < 0:
        # Affairs more common among couples with children → evidence against the hypothesis
        if p_value is not None and p_value < 0.05:
            response = 10
            explanation_parts.append(
                "Affairs are statistically more common among couples with children, "
                "providing evidence against the hypothesis that children decrease affairs."
            )
        else:
            response = 25
            explanation_parts.append(
                "Affairs tend to be more common among couples with children, but "
                "the statistical evidence is weak; overall this does not support "
                "a protective effect of children."
            )
    elif diff > 0:
        # Affairs more common among couples without children → some support
        if p_value is not None and p_value < 0.05:
            response = 75
            explanation_parts.append(
                "Couples without children have a significantly higher probability of "
                "having an affair, supporting the hypothesis that children decrease affairs."
            )
        else:
            response = 60
            explanation_parts.append(
                "Couples without children have a somewhat higher probability of having an affair, "
                "but the evidence is not strongly statistically significant."
            )
    else:
        # No difference in descriptive rates
        response = 50
        explanation_parts.append(
            "The prevalence of affairs is almost identical for couples with and without children, "
            "providing no clear evidence either way."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

