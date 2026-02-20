import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # According to info.json metadata:
    # - Column "age" encodes frequency of extramarital sexual intercourse in the past year
    #   on a 0–12 scale where 0 = none.
    # - Column "religiousness" is a yes/no factor answering
    #   "Are there children in the marriage?"
    affair_freq = df["age"]
    has_affair = affair_freq > 0
    has_children = (
        df["religiousness"].astype(str).str.strip().str.lower() == "yes"
    )

    data = pd.DataFrame(
        {
            "affair_freq": affair_freq,
            "has_affair": has_affair,
            "has_children": has_children,
        }
    ).dropna()

    # Group-level summaries
    group_stats = data.groupby("has_children").agg(
        mean_affair_freq=("affair_freq", "mean"),
        prop_has_affair=("has_affair", "mean"),
        n=("affair_freq", "size"),
    )

    # Counts for two-sample proportion z-test
    counts = np.array(
        [
            data.loc[data["has_children"], "has_affair"].sum(),
            data.loc[~data["has_children"], "has_affair"].sum(),
        ]
    )
    nobs = np.array(
        [
            data.loc[data["has_children"], "has_affair"].shape[0],
            data.loc[~data["has_children"], "has_affair"].shape[0],
        ]
    )

    z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

    prop_children = counts[0] / nobs[0]
    prop_no_children = counts[1] / nobs[1]
    diff_prop = prop_children - prop_no_children

    mean_children = group_stats.loc[True, "mean_affair_freq"]
    mean_no_children = group_stats.loc[False, "mean_affair_freq"]
    diff_mean = mean_children - mean_no_children

    alpha = 0.05
    if diff_prop < 0 and p_value < alpha:
        response = "Yes"
    else:
        response = "No"

    explanation_parts = [
        f"We analyzed {int(len(data))} married individuals from the provided affairs dataset.",
        "Extramarital engagement was coded as having at least one extramarital sexual encounter in the past year",
        "using the 0–12 frequency scale in column 'age' (0 = none, values > 0 = at least one affair).",
        "The presence of children in the marriage was taken from the yes/no indicator in column 'religiousness'.",
        f"Among participants with children, {int(counts[0])}/{int(nobs[0])} "
        f"({prop_children:.1%}) reported at least one affair,",
        f"compared with {int(counts[1])}/{int(nobs[1])} "
        f"({prop_no_children:.1%}) among those without children.",
        f"The difference in proportions (children minus no children) was {diff_prop:.3f}.",
        f"A two-sample z test for proportions yielded z = {z_stat:.2f} with p = {p_value:.3f}.",
        f"Mean affair-frequency scores on the 0–12 scale were {mean_children:.3f} for those with children",
        f"and {mean_no_children:.3f} for those without children (difference {diff_mean:.3f}, children minus no children).",
    ]

    if response == "Yes":
        explanation_parts.append(
            "Because participants with children showed a lower prevalence and average frequency of extramarital"
            " affairs and this difference was statistically significant at the 5% level,"
            " the data support the conclusion that having children is associated with decreased engagement"
            " in extramarital affairs."
        )
    else:
        explanation_parts.append(
            "Although we compared the prevalence and average frequency of extramarital affairs between those"
            " with and without children, the difference was not both negative and statistically significant"
            " at the 5% level; therefore, the data do not provide strong evidence that having children"
            " decreases engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(
        json.dumps(conclusion, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

