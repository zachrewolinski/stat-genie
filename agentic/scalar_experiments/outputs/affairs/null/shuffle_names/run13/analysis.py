import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the affairs dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct key variables based on info.json metadata.

    - Column 'age' in this shuffled version codes affair frequency in the
      past year: 0 = none, 1 = once, 2 = twice, 3 = 3 times,
      7 = 4–10 times, 12 = monthly/weekly/daily.
    - Column 'religiousness' contains the yes/no indicator for whether
      there are children in the marriage (\"yes\" = children present).
    """
    df = df.copy()

    # Binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Binary predictor: having children in the marriage
    df["has_children"] = (df["religiousness"].str.lower() == "yes").astype(int)

    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    """
    Analyze whether having children decreases engagement in extramarital affairs.

    Approach:
    1. Compare proportions of any affair for those with vs. without children.
    2. Fit a logistic regression: any_affair ~ has_children.
    """
    # Proportions
    grouped = df.groupby("has_children")["any_affair"].agg(["mean", "count"])
    prop_with_children = grouped.loc[1, "mean"]
    prop_without_children = grouped.loc[0, "mean"]

    # Logistic regression (unadjusted)
    X = sm.add_constant(df["has_children"])
    y = df["any_affair"]
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef_children = result.params["has_children"]
    p_value_children = result.pvalues["has_children"]
    odds_ratio_children = float(np.exp(coef_children))

    return {
        "prop_with_children": float(prop_with_children),
        "prop_without_children": float(prop_without_children),
        "coef_children": float(coef_children),
        "p_value_children": float(p_value_children),
        "odds_ratio_children": odds_ratio_children,
        "n_with_children": int(grouped.loc[1, "count"]),
        "n_without_children": int(grouped.loc[0, "count"]),
    }


def map_results_to_likert(stats: dict) -> int:
    """
    Map the statistical evidence to a 0–100 Likert response where
    higher values mean stronger evidence that having children
    decreases engagement in extramarital affairs.
    """
    coef = stats["coef_children"]
    p_val = stats["p_value_children"]

    # Primary decision based on sign and significance of the coefficient
    if p_val < 0.01 and coef < 0:
        return 90
    if p_val < 0.05 and coef < 0:
        return 75
    if p_val < 0.1 and coef < 0:
        return 60

    # No statistically reliable evidence that children reduce affairs.
    # If anything, positive or null effect should yield a low score.
    if p_val < 0.05 and coef > 0:
        return 10

    # Non-significant and near-zero effect: essentially no evidence
    # for a protective effect of children.
    return 20


def build_explanation(stats: dict, response: int) -> str:
    """
    Build a human-readable explanation summarizing the evidence.
    """
    prop_with = stats["prop_with_children"]
    prop_without = stats["prop_without_children"]
    coef = stats["coef_children"]
    p_val = stats["p_value_children"]
    or_children = stats["odds_ratio_children"]
    n_with = stats["n_with_children"]
    n_without = stats["n_without_children"]

    direction = (
        "lower" if coef < 0 else "higher"
        if coef > 0
        else "about the same"
    )

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"The dataset contains 601 married individuals, of whom {n_with} report having children "
        f"and {n_without} report not having children. Using the column that codes the frequency "
        "of extramarital sexual intercourse in the past year, I created a binary outcome "
        "indicating whether the person had any affair (non-zero frequency) and a binary predictor "
        "for having children in the marriage.\n\n"
        f"In the raw data, the proportion with at least one affair is "
        f"{prop_with:.3f} among those with children and {prop_without:.3f} among those without children. "
        "To formally test the relationship, I fit an unadjusted logistic regression model with "
        "any affair as the outcome and the children indicator as the sole predictor. "
        f"The coefficient for having children is {coef:.3f}, corresponding to an odds ratio of "
        f"{or_children:.3f}. The associated p-value is {p_val:.4f}.\n\n"
    )

    if p_val < 0.05 and coef < 0:
        explanation += (
            "This negative and statistically significant coefficient indicates that, in this sample, "
            "having children is associated with a lower likelihood of engaging in at least one "
            "extramarital affair over the past year. The Likert-scale response reflects reasonably strong "
            "evidence for a protective effect of children on affair engagement."
        )
    elif p_val < 0.05 and coef > 0:
        explanation += (
            "This positive and statistically significant coefficient indicates that, in this sample, "
            "having children is associated with a higher likelihood of engaging in at least one "
            "extramarital affair over the past year. This goes in the opposite direction of the question's "
            "hypothesis that children decrease affairs, so the Likert-scale response is very low."
        )
    else:
        explanation += (
            "Although the estimated effect of having children implies "
            f"{direction} odds of an affair, the p-value is not small enough to treat this pattern as "
            "statistically reliable evidence. In other words, after accounting for sampling variability, "
            "there is no strong support in this dataset for the idea that simply having children "
            "meaningfully decreases engagement in extramarital affairs. The Likert-scale response therefore "
            "reflects a primarily 'No' answer with only weak evidence for any protective effect."
        )

    explanation += (
        f"\n\nOverall Likert response (0 = strong 'No', 100 = strong 'Yes'): {response}."
    )

    return explanation


def main() -> None:
    csv_path = Path("affairs.csv")
    df = load_data(csv_path)
    df_prepared = prepare_variables(df)
    stats = analyze_relationship(df_prepared)
    response = map_results_to_likert(stats)
    explanation = build_explanation(stats, response)

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON to conclusion.txt
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

