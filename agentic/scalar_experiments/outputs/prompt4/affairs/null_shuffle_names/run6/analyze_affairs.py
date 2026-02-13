import json
from pathlib import Path

import pandas as pd


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # According to info.json metadata, the `age` column actually encodes
    # frequency of extramarital sexual intercourse, and the `religiousness`
    # column is a yes/no indicator for whether there are children.
    df = df.copy()
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Affair engagement: any non-zero value in the frequency column
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Basic group statistics by children status
    grouped = df.groupby("has_children")["age"]
    mean_freq = grouped.mean()
    prop_any_affair = df.groupby("has_children")["any_affair"].mean()

    # Compute simple effect sizes
    mean_diff = float(mean_freq.loc[1] - mean_freq.loc[0])
    prop_diff = float(prop_any_affair.loc[1] - prop_any_affair.loc[0])

    # Construct an interpretable explanation
    n_with_children = int(df["has_children"].sum())
    n_without_children = int((1 - df["has_children"]).sum())

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "In the provided dataset, the `age` column encodes the frequency of extramarital sexual "
        "intercourse in the past year, while the `religiousness` column (values 'yes'/'no') "
        "actually indicates whether there are children in the marriage."
    )
    explanation_lines.append(
        f"There are {n_with_children} respondents with children and "
        f"{n_without_children} respondents without children."
    )
    explanation_lines.append(
        "Treating any non-zero affair frequency as indicating at least one extramarital affair, "
        "I compared both the average affair frequency and the proportion with any affair "
        "between couples with and without children."
    )
    explanation_lines.append(
        f"Average affair-frequency score (higher = more frequent affairs): "
        f"with children = {mean_freq.loc[1]:.3f}, without children = {mean_freq.loc[0]:.3f} "
        f"(difference = {mean_diff:.3f}, children minus no-children)."
    )
    explanation_lines.append(
        f"Proportion with any extramarital affair in the past year: "
        f"with children = {prop_any_affair.loc[1]:.3f}, "
        f"without children = {prop_any_affair.loc[0]:.3f} "
        f"(difference = {prop_diff:.3f}, children minus no-children)."
    )

    # Interpret direction of effect: negative differences would support a "decrease" claim.
    if mean_diff < 0 and prop_diff < 0:
        qualitative = (
            "Both the mean affair-frequency score and the probability of having any affair are "
            "lower for respondents with children than for those without."
        )
        response_score = 75
    elif mean_diff > 0 and prop_diff > 0:
        qualitative = (
            "Both the mean affair-frequency score and the probability of having any affair are "
            "higher for respondents with children than for those without."
        )
        response_score = 25
    else:
        qualitative = (
            "The indicators do not point consistently in the same direction, suggesting at best a "
            "weak and ambiguous relationship between children and extramarital affairs."
        )
        response_score = 50

    explanation_lines.append(qualitative)

    explanation_lines.append(
        "Given the direction and magnitude of these differences, I translate the evidence into "
        "a 0–100 Likert score where 0 = strong 'No' and 100 = strong 'Yes' to the hypothesis "
        "that having children decreases extramarital affairs. Higher values mean stronger "
        "evidence that children are associated with fewer affairs."
    )
    explanation_lines.append(
        f"On this scale, I assign a score of {response_score}, reflecting the overall strength "
        "and consistency of the observed differences."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": int(response_score), "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

