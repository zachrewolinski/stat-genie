import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Ensure expected columns exist
    if "children" not in df.columns or "affairs" not in df.columns:
        raise ValueError("Expected 'children' and 'affairs' columns to be present in the data.")

    # Basic cleaning: drop rows with missing key fields
    df = df.dropna(subset=["children", "affairs"])

    # Create binary outcome: any affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("any_affair", "mean"),
            count=("any_affair", "size"),
        )
        .reset_index()
    )
    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))

    # Logistic regression: any_affair ~ children + controls
    # Encode children as indicator: 1 = has children, 0 = no children
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Choose a reasonable set of covariates present in the dataset
    covariates = []
    for col in ["age", "yearsmarried", "religiousness", "rating", "gender"]:
        if col in df.columns:
            covariates.append(col)

    X = df[["children_yes"] + covariates].copy()

    # Handle categorical gender via dummy coding if present
    if "gender" in X.columns:
        X = pd.get_dummies(X, columns=["gender"], drop_first=True)

    # Drop rows with missing predictors
    X = X.astype(float)
    y = df.loc[X.index, "any_affair"]

    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Logistic regression failed: {exc}")
        result = None

    effect_direction = 0.0
    p_value = 1.0

    if result is not None and "children_yes" in result.params.index:
        coef = float(result.params["children_yes"])
        p_value = float(result.pvalues["children_yes"])
        effect_direction = np.sign(coef)
        print("\nLogistic regression results for children_yes:")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  p-value:     {p_value:.4f}")
    else:
        print("\nCould not extract children effect from regression; relying on descriptives only.")

    # Decide on Yes/No: does having children decrease engagement in affairs?
    # Use both descriptive and regression evidence.
    # Compute mean difference in any_affair and number of affairs
    pivot = desc.set_index("children")
    mean_diff_any = None
    mean_diff_affairs = None

    if {"yes", "no"}.issubset({str(v).lower() for v in pivot.index}):
        # Map index robustly
        idx_map = {str(v).lower(): v for v in pivot.index}
        yes_idx = idx_map["yes"]
        no_idx = idx_map["no"]
        mean_diff_any = float(pivot.loc[yes_idx, "prop_any_affair"] - pivot.loc[no_idx, "prop_any_affair"])
        mean_diff_affairs = float(pivot.loc[yes_idx, "mean_affairs"] - pivot.loc[no_idx, "mean_affairs"])

    print("\nDifferences (children = yes minus no):")
    print(f"  Δ proportion any affair: {mean_diff_any}")
    print(f"  Δ mean number of affairs: {mean_diff_affairs}")

    response = "No"
    strength = 50
    confidence = 50
    explanation_parts = []

    if mean_diff_any is not None and mean_diff_affairs is not None:
        explanation_parts.append(
            f"On average, respondents with children have a change of {mean_diff_affairs:.3f} in the number of affairs "
            f"and a change of {mean_diff_any:.3f} in the probability of having any affair compared to those without children."
        )

    if result is not None and "children_yes" in result.params.index:
        explanation_parts.append(
            f"In a logistic regression controlling for age, years married, religiousness, rating, and gender, "
            f"the coefficient for having children (children_yes) is {result.params['children_yes']:.3f} with p-value {p_value:.3f}."
        )

    # Determine direction from descriptives
    descriptive_signal = 0
    if mean_diff_any is not None and mean_diff_affairs is not None:
        # Treat both differences; majority sign drives the descriptive signal
        signs = [np.sign(mean_diff_any), np.sign(mean_diff_affairs)]
        if signs.count(1) > signs.count(-1):
            descriptive_signal = 1
        elif signs.count(-1) > signs.count(1):
            descriptive_signal = -1

    # Combine descriptive and regression signals
    signals = []
    if descriptive_signal != 0:
        signals.append(descriptive_signal)
    if effect_direction != 0:
        signals.append(effect_direction)

    combined = np.sign(sum(signals)) if signals else 0

    # Map signals and statistical strength to answer.
    # We only answer "Yes" when there is a consistent negative association
    # and the regression provides conventional statistical significance;
    # otherwise we conclude there is no clear evidence of a decrease.
    if result is not None and "children_yes" in (result.params.index if result is not None else []):
        if p_value < 0.05 and combined < 0:
            response = "Yes"
        else:
            response = "No"
    else:
        # Without regression evidence, rely on descriptives but be conservative.
        if combined < 0:
            response = "Yes"
        else:
            response = "No"

    # Strength: magnitude of effects and significance
    effect_size = 0.0
    if mean_diff_any is not None:
        effect_size += abs(mean_diff_any)
    if mean_diff_affairs is not None:
        effect_size += abs(mean_diff_affairs) / 10.0  # scale number of affairs

    # Simple scaling: clamp to [0, 1] then to [0, 100]
    raw_strength = min(max(effect_size, 0.0), 1.0)
    strength = int(round(20 + 80 * raw_strength))

    # Confidence incorporates p-value where available
    if result is not None and "children_yes" in result.params.index:
        if p_value < 0.01:
            confidence = 90
        elif p_value < 0.05:
            confidence = 75
        elif p_value < 0.1:
            confidence = 65
        else:
            confidence = 55
    else:
        confidence = 50

    # Ensure bounds
    strength = int(max(0, min(100, strength)))
    confidence = int(max(0, min(100, confidence)))

    if response == "Yes":
        explanation_parts.append(
            "Taken together, these results suggest that having children is associated with a modest decrease in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Given the small and statistically uncertain differences, the data do not provide strong evidence that having children decreases engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write the required JSON to conclusion.txt
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")

    print("\nConclusion written to conclusion.txt:")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
