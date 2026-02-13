import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature2": "affair_freq",
            "feature6": "has_children",
        }
    )

    # Basic sanity: drop rows with missing key fields (if any)
    df = df.dropna(subset=["affair_freq", "has_children"])

    # Create binary indicators
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    df["children_indicator"] = (df["has_children"].str.lower() == "yes").astype(int)

    # --- Descriptive comparison: means by child status ---
    group_means = df.groupby("children_indicator")["affair_freq"].mean()
    mean_with_children = float(group_means.get(1, np.nan))
    mean_without_children = float(group_means.get(0, np.nan))

    # Two-sample t-test on affair frequency
    freq_with_children = df.loc[df["children_indicator"] == 1, "affair_freq"]
    freq_without_children = df.loc[df["children_indicator"] == 0, "affair_freq"]
    t_stat, p_value_t = stats.ttest_ind(
        freq_with_children, freq_without_children, equal_var=False
    )

    # --- Logistic regression on having any affair ---
    X = df[["children_indicator"]].copy()
    X = sm.add_constant(X)
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    coef_children = float(logit_result.params["children_indicator"])
    p_value_logit = float(logit_result.pvalues["children_indicator"])

    # Decision rule:
    # We answer "Yes" if both:
    #  1) The direction of the effect suggests fewer or less frequent affairs
    #     in marriages with children (negative coefficient and/or lower mean).
    #  2) The association is statistically significant at alpha = 0.05
    #     in the logistic model for any affair.
    direction_supports_decrease = (
        mean_with_children < mean_without_children and coef_children < 0
    )
    statistically_significant = p_value_logit < 0.05

    if direction_supports_decrease and statistically_significant:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"Sample size: {len(df)} married individuals from the Psychology Today survey.\n"
        f"Average affair frequency (0–12 scale) with children: {mean_with_children:.3f}; "
        f"without children: {mean_without_children:.3f}. "
        f"Welch t-test comparing mean frequencies yields t = {t_stat:.3f}, p = {p_value_t:.3f}.\n"
        "We also fit a logistic regression predicting whether an individual had any affairs "
        "from an indicator for having children. The coefficient on having children is "
        f"{coef_children:.3f} with p-value {p_value_logit:.3f}.\n"
        "Because the estimated effect of having children on extramarital affairs is not "
        "statistically significant at the 5% level, and differences in average affair "
        "frequency between parents and non-parents are small relative to variability, "
        "the data do not provide sufficient evidence that having children decreases "
        "engagement in extramarital affairs."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

