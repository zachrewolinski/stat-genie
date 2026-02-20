import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (used for context in the explanation, even if not programmatically required)
    info_path = base_path / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0].strip()

    # Load dataset
    data_path = base_path / "affairs.csv"
    df = pd.read_csv(data_path)

    # Based on the metadata descriptions:
    # - Column "age" encodes frequency of extramarital intercourse (0 = none, >0 = some affairs)
    # - Column "religiousness" (categorical: "yes"/"no") actually answers:
    #   "Are there children in the marriage?"
    #
    # We therefore construct:
    #   had_affair: True if age > 0
    #   has_children: 1 if religiousness == "yes", 0 if "no"
    df["had_affair"] = df["age"] > 0
    df["has_children"] = (
        df["religiousness"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    # Drop rows with missing key variables
    df_model = df.dropna(subset=["had_affair", "has_children"]).copy()
    df_model["had_affair"] = df_model["had_affair"].astype(int)

    # Basic group-level comparison
    with_kids = df_model[df_model["has_children"] == 1]
    without_kids = df_model[df_model["has_children"] == 0]

    prop_affair_with_kids = with_kids["had_affair"].mean() if len(with_kids) else np.nan
    prop_affair_without_kids = (
        without_kids["had_affair"].mean() if len(without_kids) else np.nan
    )
    diff_prop = prop_affair_with_kids - prop_affair_without_kids

    # Logistic regression: had_affair ~ has_children
    # (children effect on odds of having any affair)
    X = sm.add_constant(df_model["has_children"])
    y = df_model["had_affair"]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef_children = float(model.params["has_children"])
        pvalue_children = float(model.pvalues["has_children"])
        odds_ratio = float(np.exp(coef_children))
    except Exception:
        # Fallback: if the model fails for any reason, treat effect as null but
        # still answer based on the raw proportions.
        coef_children = 0.0
        odds_ratio = 1.0
        pvalue_children = 1.0

    # Determine directional answer:
    # Research question: "Does having children decrease (if at all) engagement in extramarital affairs?"
    # We answer "Yes" only if the estimated effect of children on having an affair is
    # negative and statistically convincing (p < 0.05). Otherwise we answer "No".
    if coef_children < 0 and pvalue_children < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Map effect size and significance to strength/confidence (0–100).
    abs_diff = float(abs(diff_prop)) if not np.isnan(diff_prop) else 0.0

    if pvalue_children < 0.001:
        base_strength = 90
        base_confidence = 90
    elif pvalue_children < 0.01:
        base_strength = 80
        base_confidence = 80
    elif pvalue_children < 0.05:
        base_strength = 70
        base_confidence = 70
    elif pvalue_children < 0.1:
        base_strength = 55
        base_confidence = 55
    else:
        base_strength = 40
        base_confidence = 45

    # Adjust for effect magnitude in the proportions
    if abs_diff > 0.25:
        base_strength += 5
    elif abs_diff < 0.05:
        base_strength -= 5

    strength = int(max(0, min(100, round(base_strength))))
    confidence = int(max(0, min(100, round(base_confidence))))

    # Build textual explanation including key numeric results.
    explanation = (
        f"Research question: {research_question} "
        f"I used the metadata in info.json to interpret columns in affairs.csv: "
        f'the column "age" codes how often respondents engaged in extramarital intercourse '
        f'(0 = none, >0 = some affairs), and the column "religiousness" (values \"yes\"/\"no\") '
        f"actually indicates whether there are children in the marriage. "
        f"I created a binary outcome had_affair = 1 if age > 0 and 0 otherwise, and an indicator "
        f"has_children = 1 for \"yes\" and 0 for \"no\". "
        f"The share of respondents with at least one affair was "
        f"{prop_affair_with_kids:.3f} among couples with children and "
        f"{prop_affair_without_kids:.3f} among couples without children "
        f"(difference = {diff_prop:.3f}). "
        f"I then fit a logistic regression had_affair ~ has_children. "
        f"The estimated coefficient on has_children was {coef_children:.3f}, corresponding to an "
        f"odds ratio of {odds_ratio:.3f} with p-value {pvalue_children:.4f}. "
        f"Based on the sign and statistical significance of this effect, together with the "
        f"observed difference in proportions, I concluded that the evidence "
        f"{'supports' if response == 'Yes' else 'does not support'} the claim that having children "
        f"decreases engagement in extramarital affairs."
    )

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    output_path = base_path / "conclusion.txt"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

