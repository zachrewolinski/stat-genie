import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_likert_response(diff_prop: float, coef_children: float, p_children: float) -> int:
    """
    Map evidence strength and direction to a 0–100 Likert response.

    0  = strong "No" (children do not decrease affairs, possibly increase)
    50 = "No clear effect / mixed"
    100 = strong "Yes" (children clearly decrease affairs)
    """
    # If we cannot compute a meaningful difference, stay neutral.
    if np.isnan(diff_prop):
        return 50

    # Direction from raw proportions: positive if "no children" group has more affairs.
    direction_raw = 1 if diff_prop > 0 else (-1 if diff_prop < 0 else 0)

    # Direction from regression: negative coefficient -> children associated with fewer affairs.
    if np.isnan(coef_children):
        direction_logit = 0
    else:
        if coef_children < 0:
            direction_logit = 1
        elif coef_children > 0:
            direction_logit = -1
        else:
            direction_logit = 0

    # Combine directions.
    if direction_raw == direction_logit:
        direction = direction_raw
    else:
        # If they disagree or only one is informative, treat as weaker evidence.
        if direction_raw == 0 and direction_logit == 0:
            direction = 0
        elif direction_raw == 0:
            direction = direction_logit
        elif direction_logit == 0:
            direction = direction_raw
        else:
            # Conflicting signs: treat as no clear directional evidence.
            direction = 0

    # Scale effect size based on difference in proportions.
    effect_strength = abs(diff_prop)  # in [0, 1]
    # A 20 percentage point difference is treated as "strong" evidence.
    magnitude = min(1.0, effect_strength / 0.20)

    # Weight by statistical significance.
    if np.isnan(p_children):
        sig_weight = 0.3
    elif p_children < 0.01:
        sig_weight = 1.0
    elif p_children < 0.05:
        sig_weight = 0.8
    elif p_children < 0.10:
        sig_weight = 0.5
    else:
        sig_weight = 0.3

    max_adjustment = 40  # so scores lie roughly between 10 and 90
    adjustment = int(round(max_adjustment * magnitude * sig_weight))

    if direction > 0:
        response = 50 + adjustment
    elif direction < 0:
        response = 50 - adjustment
    else:
        response = 50

    # Clip to valid range and return as plain int.
    return int(max(0, min(100, response)))


def analyze_affairs() -> Dict[str, object]:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital intercourse in past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode key predictors.
    df["children_yes"] = (df["children"] == "yes").astype(int)
    df["gender_male"] = (df["gender"] == "male").astype(int)

    # Group-wise summary by children status.
    grouped = df.groupby("children").agg(
        mean_affairs=("affairs", "mean"),
        median_affairs=("affairs", "median"),
        prop_any_affair=("any_affair", "mean"),
        count=("affairs", "size"),
    )

    # Ensure both levels exist.
    if set(grouped.index) >= {"yes", "no"}:
        mean_yes = grouped.loc["yes", "mean_affairs"]
        mean_no = grouped.loc["no", "mean_affairs"]
        prop_yes = grouped.loc["yes", "prop_any_affair"]
        prop_no = grouped.loc["no", "prop_any_affair"]
        n_yes = int(grouped.loc["yes", "count"])
        n_no = int(grouped.loc["no", "count"])
    else:
        # Fallback in pathological cases; treat diffs as NaN.
        mean_yes = mean_no = np.nan
        prop_yes = prop_no = np.nan
        n_yes = n_no = 0

    diff_prop = prop_no - prop_yes if not np.isnan(prop_no) and not np.isnan(prop_yes) else np.nan

    # Logistic regression: any_affair ~ children + controls.
    # Controls: age, yearsmarried, religiousness, rating, gender.
    cols = ["children_yes", "age", "yearsmarried", "religiousness", "rating", "gender_male"]
    X = df[cols].copy()
    y = df["any_affair"].astype(int)
    X = sm.add_constant(X, has_constant="add")

    coef_children = np.nan
    p_children = np.nan
    odds_ratio = np.nan

    try:
        logit_model = sm.Logit(y, X, missing="drop")
        result = logit_model.fit(disp=False)
        coef_children = float(result.params.get("children_yes", np.nan))
        p_children = float(result.pvalues.get("children_yes", np.nan))
        odds_ratio = float(np.exp(coef_children)) if not np.isnan(coef_children) else np.nan
    except Exception:
        # If the model fails for any reason, we still rely on descriptive stats.
        coef_children = np.nan
        p_children = np.nan
        odds_ratio = np.nan

    response = compute_likert_response(diff_prop, coef_children, p_children)

    direction_text = ""
    if not np.isnan(diff_prop):
        if diff_prop > 0:
            direction_text = (
                "respondents without children have a higher probability of reporting any affair "
                "than respondents with children"
            )
        elif diff_prop < 0:
            direction_text = (
                "respondents with children have a higher probability of reporting any affair "
                "than respondents without children"
            )
        else:
            direction_text = "the probability of reporting any affair is nearly identical across groups"

    # Build natural-language explanation.
    explanation_parts = []
    explanation_parts.append(
        "The research question is whether having children decreases engagement in extramarital affairs."
    )
    explanation_parts.append(
        f"In this dataset of {len(df)} married individuals, {n_yes} report having children and {n_no} report not having children."
    )

    if not np.isnan(mean_yes) and not np.isnan(mean_no):
        explanation_parts.append(
            "Affair frequency is coded so that 0 means no extramarital intercourse in the past year and higher "
            f"values indicate more frequent affairs. The mean affair score is {mean_yes:.2f} for respondents with "
            f"children and {mean_no:.2f} for respondents without children."
        )

    if not np.isnan(prop_yes) and not np.isnan(prop_no):
        explanation_parts.append(
            f"The proportion reporting at least one extramarital affair in the past year is {prop_yes:.1%} "
            f"for those with children and {prop_no:.1%} for those without children; thus, {direction_text}."
        )

    if not np.isnan(coef_children):
        effect_dir = "lower" if coef_children < 0 else "higher"
        signif_text = (
            "statistically significant at the 5% level"
            if not np.isnan(p_children) and p_children < 0.05
            else "not statistically significant at conventional levels"
        )
        explanation_parts.append(
            "To account for other factors, I fit a logistic regression model for having any affair as a function "
            "of children status, age, years married, religiousness, marital satisfaction rating, and gender. "
            f"The coefficient on the 'has children' indicator is {coef_children:.3f}, corresponding to an odds ratio "
            f"of {odds_ratio:.2f} (p-value {p_children:.3f}), meaning that, holding these other variables fixed, "
            f"having children is associated with {effect_dir} odds of reporting an affair, and this effect is {signif_text}."
        )
    else:
        explanation_parts.append(
            "A logistic regression model with children status and basic controls could not be reliably estimated, "
            "so the conclusion relies on descriptive comparisons between groups."
        )

    if response > 50:
        conclusion_sentence = (
            "Overall, the balance of evidence suggests that having children is associated with somewhat less "
            "engagement in extramarital affairs in this sample, though the effect is modest."
        )
    elif response < 50:
        conclusion_sentence = (
            "Overall, the data do not support the claim that having children decreases engagement in extramarital "
            "affairs; if anything, the association points in the opposite direction or is too weak to be conclusive."
        )
    else:
        conclusion_sentence = (
            "Overall, the data provide no clear evidence that having children meaningfully decreases engagement in "
            "extramarital affairs."
        )

    explanation_parts.append(
        f"On a 0–100 Likert scale where 0 represents a strong 'No' and 100 a strong 'Yes' to the statement "
        f"\"having children decreases engagement in extramarital affairs\", I summarize this evidence with a score "
        f"of {response}."
    )
    explanation_parts.append(conclusion_sentence)

    explanation = " ".join(explanation_parts)

    return {"response": int(response), "explanation": explanation}


def main() -> None:
    result = analyze_affairs()
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    main()

