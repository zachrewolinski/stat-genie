import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # feature2 encodes frequency of extramarital affairs in the last year
    # Define a binary indicator of any extramarital activity.
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # feature6 indicates whether there are children in the marriage (yes/no).
    df["children"] = df["feature6"].astype("category")

    # Simple descriptive rates by children status
    rates = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "affair_rate"})
    )

    # Two-sample proportion z-test for difference in affair rates
    counts = rates["sum"].to_numpy()
    nobs = rates["count"].to_numpy()
    # statsmodels proportion_ztest expects [count_without_children, count_with_children]
    # Ensure ordering aligns with categories (no, yes) if both are present.
    if set(df["children"].cat.categories) == {"no", "yes"}:
        counts = np.array(
            [
                rates.loc["no", "sum"],
                rates.loc["yes", "sum"],
            ]
        )
        nobs = np.array(
            [
                rates.loc["no", "count"],
                rates.loc["yes", "count"],
            ]
        )

    zstat, pval = sm.stats.proportions_ztest(count=counts, nobs=nobs)

    # Also estimate a logistic regression controlling for demographics
    df["gender"] = df["feature3"].astype("category")
    formula = (
        "any_affair ~ C(children) + C(gender) + feature4 + feature5 "
        "+ feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=0)
    children_coef = logit_model.params.get("C(children)[T.yes]", np.nan)
    children_pval = logit_model.pvalues.get("C(children)[T.yes]", np.nan)

    # Map evidence to Likert 0–100 where high means
    # stronger evidence that having children decreases affairs.
    # We look at direction (sign) and p-values from both tests.
    direction_support = 0.0
    sig_support = 0.0
    num_sources = 0

    # Proportion test: lower affair rate with children supports the hypothesis.
    if set(df["children"].cat.categories) == {"no", "yes"}:
        rate_no = rates.loc["no", "affair_rate"]
        rate_yes = rates.loc["yes", "affair_rate"]
        if rate_yes < rate_no:
            direction_support += 1
        elif rate_yes > rate_no:
            direction_support -= 1
        num_sources += 1
        if not np.isnan(pval):
            if pval < 0.01:
                sig_support += 2
            elif pval < 0.05:
                sig_support += 1

    # Logistic regression: negative coefficient for children supports hypothesis.
    if not np.isnan(children_coef):
        if children_coef < 0:
            direction_support += 1
        elif children_coef > 0:
            direction_support -= 1
        num_sources += 1
        if not np.isnan(children_pval):
            if children_pval < 0.01:
                sig_support += 2
            elif children_pval < 0.05:
                sig_support += 1

    # Convert these heuristics to a Likert score.
    # Start at neutral 50; positive evidence shifts upward, negative downward.
    score = 50
    if num_sources > 0:
        score += int(15 * direction_support) + int(10 * sig_support)
    score = max(0, min(100, score))

    # Build explanation text summarizing key statistics and model results.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )

    if set(df["children"].cat.categories) == {"no", "yes"}:
        rate_no = rates.loc["no", "affair_rate"]
        rate_yes = rates.loc["yes", "affair_rate"]
        explanation_lines.append(
            f"Observed affair prevalence (any in past year): "
            f"{rate_no:.3f} without children (n={rates.loc['no', 'count']}), "
            f"{rate_yes:.3f} with children (n={rates.loc['yes', 'count']})."
        )
        explanation_lines.append(
            f"A two-sample z-test for difference in proportions gives p-value={pval:.4g}."
        )

    if not np.isnan(children_coef):
        explanation_lines.append(
            "A logistic regression of any affair on children, gender, age, years married, "
            "religiousness, education, occupation, and marital satisfaction was estimated."
        )
        explanation_lines.append(
            f"The coefficient for having children is {children_coef:.3f} "
            f"with p-value={children_pval:.4g}."
        )

    if score >= 60:
        overall = (
            "Overall, the data provide evidence that having children is associated with "
            "a lower likelihood of extramarital affairs."
        )
    elif score <= 40:
        overall = (
            "Overall, the data do not provide convincing evidence that having children "
            "decreases the likelihood of extramarital affairs."
        )
    else:
        overall = (
            "Overall, the evidence on whether having children decreases extramarital "
            "affairs is mixed or weak."
        )
    explanation_lines.append(overall)

    explanation_lines.append(
        f"The Likert-scale response (0=strong 'No', 100=strong 'Yes') is set to {score} "
        f"to reflect this balance of effect direction and statistical significance."
    )

    conclusion = {
        "response": int(score),
        "explanation": " ".join(explanation_lines),
    }

    # Write to conclusion.txt in the required JSON-only format.
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

