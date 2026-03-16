import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # feature2: frequency of extramarital intercourse in past year
    # feature6: children in marriage (yes/no)
    # Create binary indicator of any affair
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic group summaries
    grouped = df.groupby("feature6")["feature2"]
    mean_freq = grouped.mean()

    grouped_any = df.groupby("feature6")["any_affair"]
    prop_any = grouped_any.mean()

    # Simple difference in means and proportions with regression-based tests
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    # Linear regression for affair frequency
    lin_model = ols("feature2 ~ children_yes", data=df).fit()
    lin_p = lin_model.pvalues.get("children_yes", np.nan)
    lin_coef = lin_model.params.get("children_yes", np.nan)

    # Logistic regression for any affair
    logit_model = logit("any_affair ~ children_yes", data=df).fit(disp=False)
    logit_p = logit_model.pvalues.get("children_yes", np.nan)
    logit_coef = logit_model.params.get("children_yes", np.nan)
    odds_ratio = float(np.exp(logit_coef))

    # Map evidence to Likert scale:
    # We care about whether having children decreases engagement in affairs.
    # children_yes coefficient < 0 and OR < 1 support "Yes".
    # We weight strength by p-values and effect sizes.
    response_score = 50

    if np.isfinite(logit_p) and np.isfinite(logit_coef):
        if logit_p < 0.001 and logit_coef < 0:
            response_score = 90
        elif logit_p < 0.01 and logit_coef < 0:
            response_score = 80
        elif logit_p < 0.05 and logit_coef < 0:
            response_score = 70
        elif logit_p < 0.05 and logit_coef > 0:
            response_score = 30
        elif logit_p < 0.1 and logit_coef < 0:
            response_score = 60
        elif logit_p < 0.1 and logit_coef > 0:
            response_score = 40
        else:
            response_score = 50

    # Clip to [0, 100] and convert to int
    response_score = int(min(max(response_score, 0), 100))

    explanation = {
        "research_question": "Does having children decrease (if at all) the engagement in extramarital affairs?",
        "data_used": {
            "num_rows": int(df.shape[0]),
            "variables": ["feature2 (affair frequency)", "feature6 (children yes/no)"],
        },
        "descriptive_stats": {
            "mean_affair_frequency_by_children": {
                str(k): float(v) for k, v in mean_freq.to_dict().items()
            },
            "proportion_any_affair_by_children": {
                str(k): float(v) for k, v in prop_any.to_dict().items()
            },
        },
        "models": {
            "linear_regression_feature2_on_children": {
                "coef_children_yes": float(lin_coef),
                "p_value_children_yes": float(lin_p),
            },
            "logistic_regression_any_affair_on_children": {
                "coef_children_yes": float(logit_coef),
                "odds_ratio_children_yes": float(odds_ratio),
                "p_value_children_yes": float(logit_p),
            },
        },
        "interpretation": (
            "Negative coefficients and odds ratios below 1 for 'children_yes' indicate that, "
            "controlling only for children status, individuals with children tend to have lower "
            "engagement in extramarital affairs. The p-values quantify the statistical evidence "
            "for this association, and the chosen Likert-scale response reflects both the "
            "direction and strength of this evidence."
        ),
    }

    result = {"response": response_score, "explanation": json.dumps(explanation)}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

