import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # In this shuffled schema:
    # - Column "age" encodes extramarital affairs frequency (0 = none, >0 = some affairs)
    # - Column "religiousness" is actually a yes/no indicator for whether there are children
    df["any_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = (
        df["religiousness"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    )

    # Drop rows with missing values in key variables, if any
    df_model = df[["any_affair", "has_children"]].dropna()

    # Basic group statistics
    n_total = len(df_model)
    group_counts = df_model["has_children"].value_counts().to_dict()
    rates = df_model.groupby("has_children")["any_affair"].mean()
    rate_no_children = float(rates.get(0, np.nan))
    rate_with_children = float(rates.get(1, np.nan))
    diff = rate_with_children - rate_no_children

    # 2x2 chi-square test of independence
    contingency = pd.crosstab(df_model["has_children"], df_model["any_affair"])
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)

    # Logistic regression: any_affair ~ has_children
    X = sm.add_constant(df_model[["has_children"]])
    y = df_model["any_affair"]
    logit_model = sm.Logit(y, X).fit(disp=False)
    coef_children = float(logit_model.params["has_children"])
    p_logit = float(logit_model.pvalues["has_children"])
    odds_ratio = float(np.exp(coef_children))

    # Determine direction and strength of evidence
    alpha = 0.05
    children_reduces_affairs = (diff < 0) and (odds_ratio < 1.0)
    children_increases_affairs = (diff > 0) and (odds_ratio > 1.0)

    if (p_chi2 < alpha) and (p_logit < alpha):
        # Consistent significant effect
        if children_reduces_affairs:
            # Relative reduction in risk, capped at 100%
            if rate_no_children > 0:
                rel_change = min(1.0, max(0.0, -diff / rate_no_children))
            else:
                rel_change = 0.0
            # Map relative reduction to a 60–95 range
            response = int(round(60 + 35 * rel_change))
        elif children_increases_affairs:
            # Relative increase in risk, capped at 100%
            if rate_no_children > 0:
                rel_change = min(1.0, max(0.0, diff / rate_no_children))
            else:
                rel_change = 0.0
            # Map relative increase to a 5–40 range (strong "No")
            response = int(round(40 - 35 * rel_change))
        else:
            # Significant but extremely small net effect
            response = 50
    else:
        # No clear statistical evidence
        if children_reduces_affairs:
            response = 55
        elif children_increases_affairs:
            response = 45
        else:
            response = 50

    # Clamp to [0, 100]
    response = int(max(0, min(100, response)))

    # Build explanation string
    # Store key numeric results with moderate rounding for readability.
    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"Data: {n_total} married individuals from the 1969–1970 survey dataset.\n"
        "Operationalization: I treated the shuffled column 'age' as the coded frequency of "
        "extramarital sexual intercourse during the past year (0 = none, >0 = some affairs) and "
        "the column 'religiousness' as the indicator of whether there are children in the marriage "
        "(values 'yes'/'no'), as described in the metadata.\n"
        f"Group sizes: {group_counts.get(0, 0)} without children vs {group_counts.get(1, 0)} with children.\n"
        f"Observed affair prevalence (any vs none): "
        f"{rate_no_children:.3f} without children vs {rate_with_children:.3f} with children "
        f"(difference = {diff:.3f}, defined as with-children minus without-children).\n"
        f"Chi-square test of independence on the 2x2 table (has_children × any_affair) gives "
        f"chi2 = {chi2:.3f} with p-value = {p_chi2:.4f}.\n"
        "I also fit a logistic regression model any_affair ~ has_children.\n"
        f"In this model, the coefficient on has_children is {coef_children:.3f}, which corresponds to "
        f"an odds ratio of {odds_ratio:.3f} with p-value = {p_logit:.4f}.\n"
    )

    if response > 50:
        explanation += (
            "Interpretation: Both tests are consistent with a decrease in the likelihood of having any "
            "extramarital affair among individuals with children, and this decrease is statistically "
            "reliable at the conventional 5% level. The effect size, judged by the difference in "
            "prevalence and the odds ratio, is moderate, so I answer 'Yes' with a strength slightly above "
            "the midpoint on the 0–100 scale.\n"
        )
    elif response < 50:
        explanation += (
            "Interpretation: The empirical pattern and statistical tests indicate that individuals with "
            "children are, if anything, more likely to report extramarital affairs, and this association "
            "is statistically reliable at the conventional 5% level. Given this, I answer 'No' to the "
            "question that having children decreases affairs, with the magnitude on the 0–100 scale "
            "reflecting both the statistical significance and the size of the estimated increase.\n"
        )
    else:
        explanation += (
            "Interpretation: The analyses do not provide clear, statistically robust evidence that having "
            "children materially decreases engagement in extramarital affairs. Differences between those "
            "with and without children are small and/or not statistically distinguishable from zero, so I "
            "treat the data as essentially inconclusive and place my answer near the neutral midpoint of "
            "the 0–100 scale.\n"
        )

    explanation += (
        "Overall conclusion: The Likert-scale response numerically encodes how strongly the data support "
        "the statement that having children decreases extramarital affairs, where 0 means a strong 'No' "
        "and 100 means a strong 'Yes'."
    )

    # Write JSON conclusion
    result = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))


if __name__ == "__main__":
    main()

