import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome: any extramarital affair in past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)
    df["has_children"] = (df["feature6"] == "yes").astype(int)

    # Group-level summaries
    with_children = df[df["has_children"] == 1]
    without_children = df[df["has_children"] == 0]

    n_with = len(with_children)
    n_without = len(without_children)

    rate_with = with_children["has_affair"].mean()
    rate_without = without_children["has_affair"].mean()

    freq_with = with_children["feature2"].mean()
    freq_without = without_children["feature2"].mean()

    rate_diff = rate_without - rate_with
    freq_diff = freq_without - freq_with

    # Logistic regression controlling for key covariates
    # feature3: gender (categorical)
    # feature4: age (numeric, binned)
    # feature5: years married (numeric, binned)
    # feature7: religiousness (numeric)
    # feature8: education (numeric)
    # feature9: occupation (numeric)
    # feature10: marriage rating (numeric)
    formula = (
        "has_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # In the parameterization used by patsy, C(feature6)[T.yes] is the effect
    # of having children relative to the baseline of no children.
    child_param_name = "C(feature6)[T.yes]"
    if child_param_name in logit_model.params:
        coef_children = float(logit_model.params[child_param_name])
        pval_children = float(logit_model.pvalues[child_param_name])
    else:
        # Fallback in unlikely case the name is different
        coef_children = float("nan")
        pval_children = float("nan")

    # Determine directional answer
    # Positive rate_diff/freq_diff => fewer affairs among those with children.
    evidence_children_reduce = (rate_diff > 0) and (freq_diff > 0)
    evidence_children_increase = (rate_diff < 0) and (freq_diff < 0)

    significant_negative = (coef_children < 0) and (pval_children < 0.05)
    significant_positive = (coef_children > 0) and (pval_children < 0.05)

    if evidence_children_reduce and significant_negative:
        response = "Yes"
    elif evidence_children_increase and significant_positive:
        response = "No"
    else:
        # If the effect is weak, mixed, or not statistically clear,
        # we answer "No" to the question of whether children *decrease* affairs.
        response = "No"

    # Map statistical evidence to a 0–100 strength score
    # focusing on the combination of group differences and p-value.
    abs_rate_diff = abs(rate_diff)
    abs_freq_diff = abs(freq_diff)

    # Basic effect magnitude heuristic
    effect_score = min(100.0, (abs_rate_diff * 400) + (abs_freq_diff * 40))

    # P-value contribution (smaller p => stronger evidence)
    if pval_children <= 0.001:
        pval_score = 40.0
    elif pval_children <= 0.01:
        pval_score = 30.0
    elif pval_children <= 0.05:
        pval_score = 20.0
    elif pval_children <= 0.1:
        pval_score = 10.0
    else:
        pval_score = 5.0

    raw_strength = effect_score + pval_score
    strength = max(0, min(100, int(round(raw_strength))))

    # Confidence reflects statistical significance and model robustness
    if pval_children <= 0.001:
        confidence = 85
    elif pval_children <= 0.01:
        confidence = 80
    elif pval_children <= 0.05:
        confidence = 70
    elif pval_children <= 0.1:
        confidence = 55
    else:
        confidence = 40

    # Ensure that if evidence is mixed for the chosen response,
    # we dampen strength and confidence somewhat.
    if response == "Yes" and not evidence_children_reduce:
        strength = min(strength, 40)
        confidence = min(confidence, 50)
    if response == "No" and evidence_children_reduce and not significant_negative:
        strength = min(strength, 50)
        confidence = min(confidence, 55)

    # Build explanation text
    rate_with_pct = rate_with * 100
    rate_without_pct = rate_without * 100

    explanation_parts = [
        "Using data on 601 first-marriage respondents, "
        "I examined whether having children is associated with lower engagement "
        "in extramarital affairs over the past year.",
        f"Among respondents with children (n={n_with}), "
        f"{rate_with_pct:.1f}% reported at least one extramarital encounter, "
        f"with an average affair-frequency score of {freq_with:.2f}.",
        f"Among respondents without children (n={n_without}), "
        f"{rate_without_pct:.1f}% reported at least one encounter, "
        f"with an average affair-frequency score of {freq_without:.2f}.",
    ]

    if not pd.isna(coef_children) and not pd.isna(pval_children):
        odds_ratio = float(pd.np.exp(coef_children)) if hasattr(pd, "np") else None
        if odds_ratio is None:
            effect_sentence = (
                "I then fit a logistic regression model for having any affair, "
                "including children in the marriage and controls for gender, age, "
                "years married, religiousness, education, occupation, and "
                "self-rated marital happiness."
            )
        else:
            effect_sentence = (
                "I then fit a logistic regression model for having any affair, "
                "including children in the marriage and controls for gender, age, "
                "years married, religiousness, education, occupation, and "
                "self-rated marital happiness; "
                f"the coefficient for having children corresponds to an odds ratio "
                f"of approximately {odds_ratio:.2f} (p-value {pval_children:.3f})."
            )
        explanation_parts.append(effect_sentence)
    else:
        explanation_parts.append(
            "A logistic regression model with children status and other covariates "
            "did not yield a stable coefficient for children, so conclusions are "
            "based primarily on group comparisons."
        )

    if response == "Yes":
        conclusion_sentence = (
            "Taken together, these results provide evidence that having children "
            "is associated with somewhat lower engagement in extramarital affairs."
        )
    else:
        conclusion_sentence = (
            "Taken together, these results do not show clear evidence that having "
            "children decreases engagement in extramarital affairs; any differences "
            "between parents and non-parents are small or statistically uncertain."
        )
    explanation_parts.append(conclusion_sentence)

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

