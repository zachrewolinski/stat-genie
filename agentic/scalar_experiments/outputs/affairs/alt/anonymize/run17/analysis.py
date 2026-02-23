import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Rename for clarity (internal use only)
    df = df.rename(
        columns={
            "feature1": "id",
            "feature2": "affairs_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Outcome: any extramarital affair in the past year (binary)
    df["affair_any"] = (df["affairs_freq"] > 0).astype(int)

    # Key predictor: children in the marriage (binary)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics: by-children differences
    desc_stats = (
        df.groupby("children")["affairs_freq", "affair_any"]
        .agg(["mean", "std", "sum", "count"])
        .reset_index()
    )

    # Logistic regression of any affair on children + covariates
    # Encode gender as indicator (male=1, female=0)
    df["male"] = (df["gender"] == "male").astype(int)

    covariates = [
        "children_yes",
        "male",
        "age",
        "years_married",
        "religiousness",
        "education",
        "occupation",
        "marriage_rating",
    ]

    X = df[covariates].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["affair_any"]

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    children_coef = float(logit_result.params["children_yes"])
    children_pval = float(logit_result.pvalues["children_yes"])
    children_or = float(np.exp(children_coef))

    # Also run a reduced model with only children as predictor
    X_simple = sm.add_constant(df[["children_yes"]], has_constant="add")
    logit_simple = sm.Logit(y, X_simple).fit(disp=False)
    children_coef_simple = float(logit_simple.params["children_yes"])
    children_pval_simple = float(logit_simple.pvalues["children_yes"])
    children_or_simple = float(np.exp(children_coef_simple))

    # Determine qualitative conclusion based on sign and significance
    # Focus on the full model but cross-check with the simple model
    significant = (children_pval < 0.05) and (children_pval_simple < 0.05)

    # Direction from both models (sign of coefficient)
    direction_full = np.sign(children_coef)
    direction_simple = np.sign(children_coef_simple)

    # Compute descriptive evidence
    group_means = (
        df.groupby("children")[["affairs_freq", "affair_any"]]
        .mean()
        .rename(columns={"affairs_freq": "mean_affairs_freq", "affair_any": "prob_affair"})
    )

    mean_with_children = float(group_means.loc["yes", "mean_affairs_freq"])
    mean_without_children = float(group_means.loc["no", "mean_affairs_freq"])
    prob_with_children = float(group_means.loc["yes", "prob_affair"])
    prob_without_children = float(group_means.loc["no", "prob_affair"])

    # Decide Likert-scale response
    if significant and direction_full < 0 and direction_simple < 0:
        # Having children is associated with *lower* odds of affairs
        # Strength calibration: combine effect size (odds ratio) and p-value
        # Smaller odds ratios and p-values closer to 0 -> stronger "Yes".
        or_strength = max(0.0, min(1.0, 1.0 - children_or))  # children_or < 1 if protective
        p_strength = max(0.0, min(1.0, (0.05 - children_pval) / 0.05))
        combined_strength = 0.5 * or_strength + 0.5 * p_strength
        response = int(round(60 + 40 * combined_strength))
        response = max(55, min(100, response))
        qualitative = "Yes"
    elif significant and direction_full > 0 and direction_simple > 0:
        # Having children associated with *higher* odds of affairs
        # For this research question, this is effectively a strong "No".
        or_strength = max(0.0, min(1.0, children_or - 1.0))  # children_or > 1 if risk factor
        p_strength = max(0.0, min(1.0, (0.05 - children_pval) / 0.05))
        combined_strength = 0.5 * or_strength + 0.5 * p_strength
        response = int(round(40 - 40 * combined_strength))
        response = max(0, min(45, response))
        qualitative = "No"
    else:
        # No consistent statistically significant evidence in either direction
        # Center the response near 50 to reflect uncertainty.
        response = 50
        qualitative = "No"

    # Build explanation text
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children in the marriage decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"The outcome was a binary indicator of any extramarital intercourse in the past year, derived from the "
        f"ordinal frequency variable (affairs_freq > 0). The key predictor was whether there are children in the "
        f"marriage (children_yes = 1 if 'yes')."
    )
    explanation_lines.append(
        "Descriptively, individuals without children showed a mean affairs frequency of "
        f"{mean_without_children:.3f} versus {mean_with_children:.3f} among those with children, and a proportion "
        f"with any affair of {prob_without_children:.3f} (no children) versus {prob_with_children:.3f} (with children)."
    )
    explanation_lines.append(
        "I fitted a logistic regression of any affair on the children indicator and covariates (gender, age, years "
        "married, religiousness, education, occupation, and self-rated marital happiness), and also a reduced model "
        "with only the children indicator."
    )
    explanation_lines.append(
        f"In the adjusted model, the coefficient for having children was {children_coef:.3f} "
        f"(odds ratio {children_or:.3f}, p-value {children_pval:.4f}); in the reduced model it was "
        f"{children_coef_simple:.3f} (odds ratio {children_or_simple:.3f}, p-value {children_pval_simple:.4f})."
    )

    if qualitative == "Yes":
        explanation_lines.append(
            "Both models showed a statistically significant negative association between having children and the "
            "likelihood of any extramarital affair, indicating that, in this sample, having children is associated "
            "with lower engagement in extramarital affairs."
        )
    elif significant:
        explanation_lines.append(
            "Both models showed a statistically significant positive association between having children and the "
            "likelihood of any extramarital affair, indicating that, in this sample, having children is associated "
            "with higher engagement in extramarital affairs rather than a decrease."
        )
    else:
        explanation_lines.append(
            "Across models, the coefficient for having children was not consistently statistically significant, so "
            "there is insufficient evidence in this dataset to conclude that having children meaningfully decreases "
            "engagement in extramarital affairs."
        )

    explanation_lines.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the research question, "
        f"I summarize the evidence as a {response} ({qualitative})."
    )

    explanation = " ".join(explanation_lines)

    # Write conclusion.json-style file as required
    conclusion = {"response": int(response), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

