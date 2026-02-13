import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Focus on the relationship between having children and any extramarital affair.
    df = df.dropna(subset=["affairs", "children"])
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    grouped = df.groupby("children")["any_affair"].agg(["sum", "count"])

    # Ensure both groups are present.
    if not {"yes", "no"}.issubset(grouped.index.astype(str)):
        raise ValueError("Expected both 'yes' and 'no' levels in 'children' column.")

    successes = np.array(
        [grouped.loc["yes", "sum"], grouped.loc["no", "sum"]], dtype=float
    )
    nobs = np.array(
        [grouped.loc["yes", "count"], grouped.loc["no", "count"]], dtype=float
    )

    prop_yes = successes[0] / nobs[0]
    prop_no = successes[1] / nobs[1]

    # One-sided test: H1: p_yes < p_no (having children lowers the probability of any affair).
    stat, pval = proportions_ztest(count=successes, nobs=nobs, alternative="smaller")

    alpha = 0.05
    if (prop_yes < prop_no) and (pval < alpha):
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"In the sample (N={int(nobs.sum())}), the proportion who reported at least one affair "
        f"is {prop_yes:.3f} among respondents with children (n={int(nobs[0])}) and "
        f"{prop_no:.3f} among respondents without children (n={int(nobs[1])}).\n"
        f"A one-sided z-test for difference in proportions (H1: p_with_children < p_without_children) "
        f"yields z = {stat:.3f} and p-value = {pval:.3f} (alpha = 0.05).\n"
        "Because this test "
        + ("is" if (prop_yes < prop_no and pval < alpha) else "is not")
        + " statistically significant, "
        + (
            "there is evidence in this dataset that having children is associated with a lower likelihood of reporting any extramarital affair."
            if (prop_yes < prop_no and pval < alpha)
            else "the data do not provide statistically reliable evidence that having children decreases engagement in extramarital affairs."
        )
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

