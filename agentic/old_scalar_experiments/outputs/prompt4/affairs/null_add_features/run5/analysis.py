import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for any extramarital affair in the last year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Binary indicator for having children (1 = yes, 0 = no)
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    # Descriptive statistics by children status
    group_affairs = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            mean_has_affair=("has_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )

    # Simple logistic regression: probability of any affair ~ children_yes
    y = df["has_affair"]
    X = sm.add_constant(df["children_yes"])
    logit_model = sm.Logit(y, X).fit(disp=False)
    beta_children = float(logit_model.params["children_yes"])
    p_children = float(logit_model.pvalues["children_yes"])

    # Compute odds for each group for interpretation
    intercept = float(logit_model.params["const"])
    logit_no_children = intercept
    logit_with_children = intercept + beta_children
    odds_no_children = np.exp(logit_no_children)
    odds_with_children = np.exp(logit_with_children)
    prob_no_children = odds_no_children / (1 + odds_no_children)
    prob_with_children = odds_with_children / (1 + odds_with_children)

    # Print a concise summary for human inspection
    print("Descriptive statistics by children status:")
    print(group_affairs.to_string(index=False))
    print("\nLogistic regression: has_affair ~ children_yes")
    print(logit_model.summary())
    print(
        f"\nEstimated probability of any affair (no children): {prob_no_children:.3f}"
    )
    print(
        f"Estimated probability of any affair (with children): {prob_with_children:.3f}"
    )
    print(f"Coefficient (children_yes): {beta_children:.3f}, p-value: {p_children:.4g}")

    # Map statistical evidence to a 0-100 Likert-style response score
    if p_children >= 0.1:
        # Effect is statistically weak / indistinguishable from zero
        response_score = 50
    elif 0.05 <= p_children < 0.1:
        response_score = 60 if beta_children < 0 else 40
    elif 0.01 <= p_children < 0.05:
        response_score = 75 if beta_children < 0 else 25
    else:  # p_children < 0.01
        response_score = 90 if beta_children < 0 else 10

    # Build a machine-readable summary of the analysis
    summary = {
        "group_affairs": group_affairs.to_dict(orient="records"),
        "beta_children": beta_children,
        "p_children": p_children,
        "prob_no_children": prob_no_children,
        "prob_with_children": prob_with_children,
        "response_score": int(response_score),
    }

    # Persist the summary so it can be used to craft the final explanation
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

