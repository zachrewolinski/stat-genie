import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_prepare_data(path: str = "affairs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # In this dataset, column names and descriptions are partially misaligned.
    # Based on the provided metadata:
    # - "age" actually encodes how often the person engaged in extramarital intercourse
    #   during the past year (0 = none, 1 = once, 2 = twice, 3 = 3 times,
    #   7 = 4–10 times, 12 = monthly/weekly/daily).
    # - "religiousness" is a yes/no indicator for whether there are children
    #   in the marriage.
    df["affair_freq"] = df["age"]
    df["had_affair"] = (df["affair_freq"] > 0).astype(int)

    # Indicator for having children in the marriage.
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Basic sanity check: drop any rows where this mapping failed.
    df = df.dropna(subset=["has_children"]).copy()
    df["has_children"] = df["has_children"].astype(int)

    # Additional controls based on metadata:
    # - "occupation" encodes age bands.
    # - "children" encodes years married.
    # - "rating" encodes religiousness level (1–5).
    # - "yearsmarried" encodes education level.
    # - "affairs" encodes marital satisfaction (1–5).
    df["gender_male"] = (df["gender"] == "male").astype(int)

    return df


def summarize_groups(df: pd.DataFrame) -> Dict[str, float]:
    grouped_mean_freq = df.groupby("has_children")["affair_freq"].mean()
    grouped_prop_affair = df.groupby("has_children")["had_affair"].mean()

    summary = {
        "mean_freq_with_children": float(grouped_mean_freq.get(1, np.nan)),
        "mean_freq_without_children": float(grouped_mean_freq.get(0, np.nan)),
        "prop_affair_with_children": float(grouped_prop_affair.get(1, np.nan)),
        "prop_affair_without_children": float(grouped_prop_affair.get(0, np.nan)),
    }

    return summary


def fit_logistic_model(df: pd.DataFrame):
    # Logistic regression: any affair (yes/no) on children indicator + controls.
    X = df[
        [
            "has_children",
            "occupation",
            "children",
            "rating",
            "yearsmarried",
            "gender_male",
            "affairs",
        ]
    ].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["had_affair"]

    model = sm.Logit(y, X).fit(disp=False)
    return model


def interpret_results(
    summary: Dict[str, float], model
) -> Dict[str, object]:
    coef_children = float(model.params["has_children"])
    p_children = float(model.pvalues["has_children"])
    odds_ratio = float(np.exp(coef_children))

    mean_with = summary["mean_freq_with_children"]
    mean_without = summary["mean_freq_without_children"]
    prop_with = summary["prop_affair_with_children"]
    prop_without = summary["prop_affair_without_children"]

    # Determine binary answer based on direction and significance.
    if coef_children < 0 and p_children < 0.05:
        response = "Yes"
        strength = 80 if p_children < 0.01 else 70
        confidence = 80 if p_children < 0.01 else 70
    elif coef_children < 0 and p_children < 0.1:
        response = "Yes"
        strength = 60
        confidence = 60
    elif coef_children < 0:
        response = "Yes"
        strength = 40
        confidence = 45
    else:
        # Coefficient is zero or positive: no evidence of a decrease.
        response = "No"
        if p_children < 0.05:
            strength = 80
            confidence = 80
        elif p_children < 0.1:
            strength = 65
            confidence = 65
        else:
            strength = 55
            confidence = 55

    explanation = (
        "I analyzed 601 first-marriage respondents from the Psychology Today survey. "
        "Using the provided metadata, I treated the 'age' column as the frequency of "
        "extramarital intercourse in the past year and the 'religiousness' column as a "
        "yes/no indicator for whether there are children in the marriage. "
        f"Individuals with children had an average affair-frequency score of "
        f"{mean_with:.2f}, compared with {mean_without:.2f} for those without children. "
        f"The proportion with at least one extramarital encounter was "
        f"{prop_with:.1%} among those with children versus {prop_without:.1%} among "
        "those without children. "
        "I then fit a logistic regression for 'any affair' (nonzero frequency) on the "
        "'has_children' indicator while controlling for age band, years married, "
        "religiousness level, education, gender, and marital satisfaction. "
        f"The estimated coefficient on 'has_children' was {coef_children:.3f} "
        f"(odds ratio {odds_ratio:.2f}, p-value {p_children:.3g}). "
    )

    if response == "Yes":
        explanation += (
            "This negative coefficient and the group-level comparison suggest that, "
            "in this sample, having children is associated with a lower likelihood of "
            "engaging in extramarital affairs, even after adjusting for other factors."
        )
    else:
        explanation += (
            "Given the sign and significance of this coefficient together with the "
            "group-level comparison, I do not find evidence that having children "
            "decreases engagement in extramarital affairs in this dataset."
        )

    return {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
    }


def main() -> None:
    df = load_and_prepare_data()
    summary = summarize_groups(df)
    model = fit_logistic_model(df)
    result = interpret_results(summary, model)

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

