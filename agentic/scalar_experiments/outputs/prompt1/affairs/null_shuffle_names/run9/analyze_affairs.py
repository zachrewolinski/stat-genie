import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json, the column named "age" actually encodes
    # how often the respondent engaged in extramarital sex in the past year.
    # Code: 0 = none, >0 = some engagement.
    affair_engagement = df["age"] > 0

    # The column named "religiousness" is described as a yes/no factor
    # answering "Are there children in the marriage?".
    has_children = df["religiousness"].astype(str).str.lower().map({"yes": True, "no": False})

    # Keep rows with clearly defined children status
    mask_valid = has_children.notna()
    df = df[mask_valid].copy()
    affair_engagement = affair_engagement[mask_valid]
    has_children = has_children[mask_valid]

    # Group counts
    n_children = int(has_children.sum())
    n_no_children = int((~has_children).sum())
    successes_children = int((affair_engagement & has_children).sum())
    successes_no_children = int((affair_engagement & ~has_children).sum())

    count = np.array([successes_children, successes_no_children])
    nobs = np.array([n_children, n_no_children])

    # Basic sanity check to avoid division by zero
    if (n_children == 0) or (n_no_children == 0):
        response = "No"
        explanation = (
            "The dataset does not contain respondents both with and without children "
            "under the encoded children variable, so it is not possible to assess whether "
            "having children decreases engagement in extramarital affairs."
        )
    else:
        # One-sided test: proportion with affairs is smaller among those with children
        stat, p_value = proportions_ztest(count, nobs, alternative="smaller")

        prop_children = successes_children / n_children
        prop_no_children = successes_no_children / n_no_children
        difference = prop_children - prop_no_children

        alpha = 0.05
        # We only answer "Yes" if the data show a lower affair rate with children
        # and this difference is statistically significant at the 5% level.
        if (difference < 0) and (p_value < alpha):
            response = "Yes"
        else:
            response = "No"

        explanation = (
            "Using the survey of 601 married individuals, I coded engagement in extramarital "
            "affairs as having any nonzero value in the column that the metadata describe as "
            "frequency of extramarital intercourse in the past year (the column named 'age' in "
            "this file). I treated the 'religiousness' column as the indicator of whether there "
            "are children in the marriage, as stated in the metadata (values 'yes' or 'no'). "
            f"Among respondents with children, {successes_children} out of {n_children} "
            f"({prop_children:.3f}) reported at least some extramarital activity, whereas among "
            f"those without children, {successes_no_children} out of {n_no_children} "
            f"({prop_no_children:.3f}) did so. I then performed a one-sided two-sample "
            "proportion z-test for the hypothesis that the proportion of people engaging in "
            "extramarital affairs is lower among those with children than among those without. "
            f"The estimated difference in proportions (children minus no children) was "
            f"{difference:.3f} with test statistic z = {stat:.3f} and p-value = {p_value:.3f}. "
            "Based on this test at the 5% significance level, I concluded "
            "that the evidence is {} that having children is associated with a lower rate of "
            "engagement in extramarital affairs."
        ).format("sufficient" if response == "Yes" else "insufficient")

    # Write conclusion.txt as a single JSON object with no extra lines
    conclusion = {"response": response, "explanation": explanation}
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
