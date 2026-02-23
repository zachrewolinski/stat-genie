import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    with info_path.open() as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0].strip()

    data = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year (binary)
    data["any_affair"] = (data["feature2"] > 0).astype(int)

    # Predictor: children in the marriage (1 = yes, 0 = no)
    data["children"] = (
        data["feature6"].astype(str).str.strip().str.lower().eq("yes")
    ).astype(int)

    # Descriptive statistics
    has_children = data["children"] == 1
    no_children = data["children"] == 0

    n_children = int(has_children.sum())
    n_no_children = int(no_children.sum())

    prop_affair_children = float(data.loc[has_children, "any_affair"].mean())
    prop_affair_no_children = float(data.loc[no_children, "any_affair"].mean())

    mean_affair_children = float(data.loc[has_children, "feature2"].mean())
    mean_affair_no_children = float(data.loc[no_children, "feature2"].mean())

    # Logistic regression: any_affair ~ children
    y = data["any_affair"].values
    X = sm.add_constant(data["children"].values)

    logit_model = sm.Logit(y, X, missing="drop")
    result = logit_model.fit(disp=False)

    coef_children = float(result.params[1])
    p_value_children = float(result.pvalues[1])
    odds_ratio = float(np.exp(coef_children))

    # Map statistical evidence to a 0–100 Likert score, where
    # higher means stronger evidence that having children DECREASES affairs.
    response = 50

    if coef_children < 0 and p_value_children < 0.05:
        if p_value_children < 0.001:
            base = 90
        elif p_value_children < 0.01:
            base = 80
        else:
            base = 70
        effect_bonus = min(10.0, abs(coef_children) * 10.0)
        response = int(round(min(100.0, base + effect_bonus)))
    elif coef_children < 0 and p_value_children < 0.1:
        response = 60
    elif coef_children < 0:
        response = 45
    elif coef_children > 0 and p_value_children < 0.05:
        # Significant evidence in the opposite direction
        if p_value_children < 0.001:
            response = 10
        elif p_value_children < 0.01:
            response = 15
        else:
            response = 20
    elif coef_children > 0 and p_value_children < 0.1:
        response = 30
    elif coef_children > 0:
        response = 40
    else:
        response = 50

    response = max(0, min(100, int(response)))

    direction = "decrease" if coef_children < 0 else "increase"

    explanation = (
        f"Research question: {question}\n\n"
        f"Data and variables:\n"
        f"- Sample size: {len(data)} currently married individuals from the Fair affairs dataset.\n"
        f"- Outcome: any extramarital intercourse in the past year "
        f"(binary indicator derived from feature2 > 0).\n"
        f"- Predictor: presence of children in the marriage (feature6; 1 = yes, 0 = no).\n\n"
        f"Descriptive evidence:\n"
        f"- Number with children: {n_children}, proportion with at least one affair: "
        f"{prop_affair_children:.3f}, mean affair frequency: {mean_affair_children:.3f}.\n"
        f"- Number without children: {n_no_children}, proportion with at least one affair: "
        f"{prop_affair_no_children:.3f}, mean affair frequency: {mean_affair_no_children:.3f}.\n\n"
        f"Model-based evidence:\n"
        f"- Logistic regression of any affair on children (no other covariates).\n"
        f"- Coefficient on children: {coef_children:.3f} (log-odds), odds ratio: {odds_ratio:.3f}, "
        f"p-value: {p_value_children:.4f}.\n\n"
        f"Interpretation:\n"
        f"- The sign of the coefficient suggests that having children tends to {direction} the engagement "
        f"in extramarital affairs (in terms of odds), but statistical significance and effect magnitude are "
        f"captured in the mapped Likert score.\n"
        f"- The 0–100 response value of {response} reflects how strongly the data support the claim that "
        "having children decreases extramarital affairs, with higher values indicating stronger evidence."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
