import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Map semantic meanings using the metadata descriptions:
    # - Column "age": frequency of extramarital sexual intercourse in past year (0 = none, >0 = some affairs)
    # - Column "religiousness": categorical yes/no indicating whether there are children in the marriage
    df["has_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop rows with missing mappings, if any
    df = df.dropna(subset=["has_affair", "has_children"])
    df["has_children"] = df["has_children"].astype(int)

    return df


def analyze_children_effect(df: pd.DataFrame):
    # Summary statistics by children status
    group = (
        df.groupby("has_children")
        .agg(
            mean_affair_indicator=("has_affair", "mean"),
            mean_affair_frequency=("age", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )

    # Logistic regression: probability of any affair as a function of having children
    y = df["has_affair"]
    X = sm.add_constant(df["has_children"])
    model = sm.Logit(y, X).fit(disp=False)

    coef = model.params["has_children"]
    p_value = model.pvalues["has_children"]
    odds_ratio = float(np.exp(coef))

    stats = {
        "group_summary": group.to_dict(orient="records"),
        "logit_coef_children": float(coef),
        "logit_p_value_children": float(p_value),
        "logit_odds_ratio_children": odds_ratio,
    }

    # Decide on directional answer
    # Negative coefficient => having children is associated with *lower* odds of any affair
    effect_is_decrease = coef < 0
    statistically_significant = p_value < 0.05

    if effect_is_decrease and statistically_significant:
        response = "Yes"
        # High confidence when direction is consistent and significant
        confidence = 90
    elif effect_is_decrease and not statistically_significant:
        response = "Yes"
        # Some evidence but weaker; moderate confidence
        confidence = 65
    else:
        # Either no effect or increase in affairs among those with children
        response = "No"
        confidence = 80 if statistically_significant else 60

    return response, confidence, stats


def build_explanation(response: str, confidence: int, stats: dict) -> str:
    group_stats = stats["group_summary"]

    # Extract human-readable group stats
    stats_text_parts = []
    for row in group_stats:
        label = "with children" if row["has_children"] == 1 else "without children"
        stats_text_parts.append(
            f"Among those {label}, the share reporting any affair was "
            f"{row['mean_affair_indicator']:.3f} and the average affair-frequency score "
            f"was {row['mean_affair_frequency']:.3f} (n={int(row['count'])})."
        )

    direction_text = (
        "a decrease in the odds of having an affair among people with children"
        if stats["logit_coef_children"] < 0
        else "no decrease (the odds are similar or higher among people with children)"
    )

    significance_text = (
        "This association is statistically significant at the 5% level "
        f"(p-value ≈ {stats['logit_p_value_children']:.3g}), "
        if stats["logit_p_value_children"] < 0.05
        else "This association is not statistically significant at the 5% level "
        f"(p-value ≈ {stats['logit_p_value_children']:.3g}), "
    )

    odds_text = (
        f"and the estimated odds ratio for having children is "
        f"{stats['logit_odds_ratio_children']:.3f}, "
        "interpreted as the multiplicative change in the odds of any affair "
        "for those with children relative to those without."
    )

    stats_text = " ".join(stats_text_parts)

    conclusion_text = (
        f"To assess whether having children decreases engagement in extramarital affairs, "
        f"I used the provided survey data where the 'age' column represents the frequency of extramarital "
        f"sexual intercourse in the past year and the 'religiousness' column indicates whether there are "
        f"children in the marriage (yes/no). I defined a binary outcome of any affair (age > 0) and fit a "
        f"logistic regression with this outcome as the dependent variable and the children indicator as the "
        f"sole predictor. {stats_text} "
        f"The regression coefficient on the children indicator corresponds to {direction_text}. "
        f"{significance_text}{odds_text} "
        f"Based on this evidence, my answer to the question "
        f"'Does having children decrease (if at all) the engagement in extramarital affairs?' is '{response}' "
        f"with a confidence score of {confidence} out of 100."
    )

    return conclusion_text


def main():
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)

    response, confidence, stats = analyze_children_effect(df)
    explanation = build_explanation(response, confidence, stats)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

