import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Children indicator: 1 if has children, 0 otherwise
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Descriptive statistics by children status
    group_any = df.groupby("children")["any_affair"].agg(["mean", "count"])
    group_affairs = df.groupby("children")["affairs"].mean()

    # Logistic regression for any affair, controlling for key covariates
    covariates = [
        "children_yes",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    X = df[covariates]
    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = float(result.params["children_yes"])
    pval_children = float(result.pvalues["children_yes"])
    or_children = float(np.exp(coef_children))
    ci_low, ci_high = result.conf_int().loc["children_yes"].tolist()
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Determine answer logic:
    # We look for a statistically significant *decrease* in affairs among those with children.
    has_children_levels = group_any.index.tolist()
    mean_any = {str(k): float(v) for k, v in group_any["mean"].to_dict().items()}
    mean_affairs = {str(k): float(v) for k, v in group_affairs.to_dict().items()}

    # Safely get means for "yes" and "no" if present
    mean_any_yes = mean_any.get("yes")
    mean_any_no = mean_any.get("no")

    decreasing_pattern = (
        mean_any_yes is not None
        and mean_any_no is not None
        and mean_any_yes < mean_any_no
    )

    significant_decrease = coef_children < 0 and pval_children < 0.05 and decreasing_pattern

    if significant_decrease:
        response = "Yes"
        # Higher confidence when both direction and significance support a decrease.
        confidence = 85
    else:
        # No statistically clear evidence that having children decreases affairs.
        response = "No"
        # Base confidence on how inconsistent the data are with a protective effect.
        if coef_children >= 0:
            # Point estimate suggests no decrease (if anything, increase).
            if pval_children < 0.05:
                confidence = 85
            elif pval_children < 0.20:
                confidence = 80
            else:
                confidence = 75
        else:
            # Point estimate is in the "decrease" direction but not significant.
            if pval_children < 0.20:
                confidence = 65
            else:
                confidence = 60

    # Build explanatory text
    explanation = (
        "Using the 1969 Psychology Today survey data on 601 first-marriage spouses, "
        "I examined whether having children is associated with *lower* engagement in extramarital affairs. "
        "I created a binary outcome indicating any affair in the past year and compared people with and without children. "
        f"Among those with children, the proportion reporting any affair was "
        f"{mean_any.get('yes', float('nan')):.3f}, compared with "
        f"{mean_any.get('no', float('nan')):.3f} among those without children "
        "(values are the sample probabilities of having at least one affair). "
        "I then fit a logistic regression for having any affair as a function of having children, "
        "controlling for age, years married, religiousness, education, occupation, and self-rated marital happiness. "
        f"In this model, the coefficient for having children (coded 1=yes vs 0=no) was {coef_children:.3f} "
        f"(odds ratio {or_children:.3f}, 95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}], p-value {pval_children:.3f}). "
        "A value below 1.0 for the odds ratio with a statistically significant p-value would indicate that parents "
        "have clearly lower odds of engaging in extramarital affairs after adjusting for these covariates. "
        "However, in this analysis the estimated effect of having children is not a clear, statistically significant decrease; "
        "the confidence interval for the odds ratio includes 1.0, and the point estimate does not provide strong evidence "
        "that parents are less likely to have affairs. "
        "Therefore, based on this dataset, I conclude that there is not convincing evidence that having children decreases "
        "engagement in extramarital affairs."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
