import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome definitions
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = (df["feature6"] == "yes").astype(int)

    # Quick descriptive stats
    desc = (
        df.groupby("children")["feature2"]
        .agg(["mean", "std", "count"])
        .rename(index={0: "no_children", 1: "children"})
    )
    print("Affair frequency by children:\n", desc, "\n", flush=True)

    # Unadjusted logistic regression: any affair ~ children
    logit_unadj = smf.logit("any_affair ~ children", data=df).fit(disp=False)
    print("Unadjusted logistic regression:\n", logit_unadj.summary(), "\n", flush=True)

    # Adjusted logistic regression including key covariates
    formula_adj = (
        "any_affair ~ children + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print("Adjusted logistic regression:\n", logit_adj.summary(), "\n", flush=True)

    # Extract children effect from adjusted model
    children_coef = logit_adj.params["children"]
    children_pval = logit_adj.pvalues["children"]

    # Also compute mean difference and a simple t-statistic
    with_children = df[df["children"] == 1]["feature2"]
    without_children = df[df["children"] == 0]["feature2"]
    mean_diff = with_children.mean() - without_children.mean()
    pooled_sd = np.sqrt(
        ((with_children.var(ddof=1) * (with_children.size - 1)) +
         (without_children.var(ddof=1) * (without_children.size - 1)))
        / (with_children.size + without_children.size - 2)
    )
    cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

    print(
        f"Mean difference (children - no_children): {mean_diff:.3f}, "
        f"Cohen's d: {cohen_d:.3f}",
        flush=True,
    )

    # Map findings to a Likert-style 0-100 response.
    # Research question: does having children decrease extramarital affairs?
    # We interpret a negative, statistically significant children coefficient
    # as evidence for a decrease; here we expect the opposite.
    if children_pval < 0.05 and children_coef < 0:
        response = 80
        qualitative = (
            "Strong evidence that having children is associated with fewer "
            "extramarital affairs, even after adjusting for covariates."
        )
    elif children_pval < 0.05 and children_coef > 0:
        response = 15
        qualitative = (
            "Statistically significant evidence that having children is "
            "associated with more, not fewer, extramarital affairs, "
            "controlling for other factors."
        )
    else:
        response = 40
        qualitative = (
            "No statistically strong evidence that having children decreases "
            "extramarital affairs; any differences are small or uncertain."
        )

    explanation = {
        "children_coef": children_coef,
        "children_pval": children_pval,
        "mean_affairs_children": float(with_children.mean()),
        "mean_affairs_no_children": float(without_children.mean()),
        "mean_difference_children_minus_no_children": float(mean_diff),
        "cohen_d": float(cohen_d) if not np.isnan(cohen_d) else None,
        "qualitative_conclusion": qualitative,
    }

    # Only the scalar and human-readable explanation go to conclusion.txt
    explanation_text = (
        "Based on logistic regression models of any extramarital affair "
        "on the presence of children, with and without adjustment for "
        "gender, age, years married, religiousness, education, "
        "occupation, and marital satisfaction, the coefficient for "
        f"having children is {children_coef:.3f} with p-value {children_pval:.3g}. "
        "Additionally, the mean coded affair frequency is "
        f"{with_children.mean():.3f} for couples with children and "
        f"{without_children.mean():.3f} for couples without children "
        f"(difference {mean_diff:.3f}, Cohen's d {cohen_d:.3f}). "
        + qualitative
    )

    conclusion = {
        "response": int(response),
        "explanation": explanation_text,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")

    # Also store the more detailed statistics for transparency, if needed.
    Path("analysis_details.json").write_text(
        json.dumps(explanation, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
