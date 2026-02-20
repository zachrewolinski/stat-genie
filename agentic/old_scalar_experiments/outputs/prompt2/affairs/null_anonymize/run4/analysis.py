import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    # Load data
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define variables
    # feature2: frequency of extramarital intercourse in past year (0 = none)
    # feature6: "yes"/"no" for children in the marriage
    df["any_affair"] = df["feature2"] > 0

    # Group-level summaries
    prop_by_children = df.groupby("feature6")["any_affair"].mean()
    freq_by_children = df.groupby("feature6")["feature2"].mean()

    prop_children_yes = float(prop_by_children.get("yes", float("nan")))
    prop_children_no = float(prop_by_children.get("no", float("nan")))

    mean_freq_children_yes = float(freq_by_children.get("yes", float("nan")))
    mean_freq_children_no = float(freq_by_children.get("no", float("nan")))

    # Contingency table and chi-square test for association between
    # children (yes/no) and any affair (yes/no)
    contingency = pd.crosstab(df["feature6"], df["any_affair"])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Determine answer based on direction and statistical evidence
    # Default assumptions
    response = "No"
    confidence = 70

    children_reduce_affairs = prop_children_yes < prop_children_no
    statistically_significant = p_value < 0.05

    if children_reduce_affairs and statistically_significant:
        # Clear evidence that parents have fewer affairs
        response = "Yes"
        confidence = 85
    elif not children_reduce_affairs and statistically_significant:
        # Evidence points the other way (or no reduction)
        response = "No"
        confidence = 85
    elif children_reduce_affairs and not statistically_significant:
        # Directionally lower but not strongly supported
        response = "No"
        confidence = 65
    else:
        # No directional reduction and no clear association
        response = "No"
        confidence = 75

    # Build explanation string with key statistics
    explanation = (
        "Using the 1969 Psychology Today survey sample (n = {n}), I compared "
        "engagement in extramarital affairs between marriages with and without children. "
        "I created a binary indicator of any affair in the past year (feature2 > 0) and "
        "cross‑tabulated it with the presence of children (feature6). "
        "In the data, the share reporting any affair was {prop_no:.1%} for couples without "
        "children and {prop_yes:.1%} for couples with children; the average affair frequency "
        "codes were {mean_no:.3f} (no children) vs {mean_yes:.3f} (with children). "
        "A chi‑square test of independence between children and any affair yielded "
        "chi² = {chi2:.3f} with p = {p_value:.4f}. "
        "Based on the direction of the group differences and this p‑value, I conclude that "
    ).format(
        n=len(df),
        prop_no=prop_children_no,
        prop_yes=prop_children_yes,
        mean_no=mean_freq_children_no,
        mean_yes=mean_freq_children_yes,
        chi2=chi2,
        p_value=p_value,
    )

    if response == "Yes":
        explanation += (
            "having children is associated with a lower rate of extramarital affairs "
            "in this sample."
        )
    else:
        explanation += (
            "there is not strong evidence that having children decreases engagement "
            "in extramarital affairs in this sample."
        )

    # Write JSON conclusion
    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

