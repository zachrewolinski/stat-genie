import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary outcome: any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children: yes=1, no=0
    df["children_bin"] = (df["children"] == "yes").astype(int)

    # Controls based on classic Fair (1978) specification
    X = df[
        [
            "children_bin",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
    ].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    children_effect = result.params["children_bin"]
    children_pvalue = result.pvalues["children_bin"]

    # Also compare raw proportions as a simple descriptive check
    prop_by_children = df.groupby("children")["any_affair"].mean()
    prop_children_yes = float(prop_by_children.get("yes", np.nan))
    prop_children_no = float(prop_by_children.get("no", np.nan))

    # Research question: Does having children decrease engagement in affairs?
    # Null we are testing against: "No, having children does not decrease affairs"
    decreases_affairs = children_effect < 0 and children_pvalue < 0.05

    if decreases_affairs:
        response = "Yes"
        base_confidence = 80
    else:
        response = "No"
        base_confidence = 75

    # Adjust confidence based on consistency with descriptive stats
    descriptive_supports_decrease = prop_children_yes < prop_children_no
    if decreases_affairs and descriptive_supports_decrease:
        confidence = min(100, base_confidence + 10)
    elif (not decreases_affairs) and (not descriptive_supports_decrease):
        confidence = min(100, base_confidence + 10)
    else:
        confidence = max(50, base_confidence - 10)

    explanation = {
        "model_type": "logistic_regression",
        "outcome": "any_affair (1 if >0 affairs in past year)",
        "key_predictor": "children (yes vs no)",
        "children_logit_coefficient": float(children_effect),
        "children_logit_pvalue": float(children_pvalue),
        "proportion_any_affair_children_yes": prop_children_yes,
        "proportion_any_affair_children_no": prop_children_no,
        "interpretation": (
            "A negative and statistically significant coefficient on children would "
            "indicate lower odds of any extramarital affair among respondents with children, "
            "after controlling for age, years married, religiousness, education, occupation, "
            "and self-rated marital happiness. The descriptive proportions compare the raw "
            "share of respondents with any affairs between those with and without children."
        ),
    }

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": json.dumps(explanation),
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

