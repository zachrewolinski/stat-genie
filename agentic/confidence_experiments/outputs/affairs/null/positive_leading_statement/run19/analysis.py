import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base = Path(__file__).parent

    info_path = base / "info.json"
    data_path = base / "affairs.csv"

    with info_path.open() as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary 1=yes, 0=no for modeling and summaries
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics: affair prevalence by children status
    group_stats = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prevalence", "count": "n"})
    )

    # Logistic regression: does having children predict lower odds of affairs,
    # adjusting for standard covariates from this dataset?
    formula = "any_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    children_coef = model.params["children_yes"]
    children_p = model.pvalues["children_yes"]
    children_or = float(np.exp(children_coef))

    # Map statistical evidence to a 0–100 Likert-style scalar where
    # high values mean "Yes, having children decreases affairs".
    #
    # Interpretation:
    # - If coefficient is negative (OR<1) and significant, this supports the hypothesis.
    # - If coefficient is near zero or positive, or not significant, this weakens it.
    #
    # We use a simple heuristic:
    # - Start at 50 (no clear evidence).
    # - Adjust down if effect is in the opposite direction or not significant.
    # - Adjust up if effect is negative and significant, scaled by magnitude and p-value.
    response = 50

    if children_p >= 0.05:
        # No statistically significant effect; keep near neutral but
        # nudge slightly toward the observed direction.
        if children_coef < 0:
            response = 45
        else:
            response = 40
    else:
        # Statistically significant effect: scale strength.
        # Stronger negative coefficients and smaller p-values move response higher.
        if children_coef < 0:
            # Negative (protective) effect
            # Cap magnitude at abs(coef)=1.0 for scaling.
            strength = min(abs(children_coef), 1.0)
            # Smaller p-values (down to 1e-6) get more weight.
            p_scaled = max(min(children_p, 0.05), 1e-6)
            evidence_weight = 1 - (p_scaled / 0.05)
            response = int(round(60 + 30 * strength * evidence_weight))
        else:
            # Significant but positive (children associated with more affairs)
            strength = min(children_coef, 1.0)
            p_scaled = max(min(children_p, 0.05), 1e-6)
            evidence_weight = 1 - (p_scaled / 0.05)
            response = int(round(40 - 30 * strength * evidence_weight))

    # Clamp to [0, 100] and cast to int
    response = int(max(0, min(100, response)))

    # Build human-readable explanation
    prev_with_children = group_stats.loc["yes", "prevalence"]
    prev_without_children = group_stats.loc["no", "prevalence"]

    direction_text = (
        "lower" if children_coef < 0 else "higher" if children_coef > 0 else "similar"
    )

    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        f"In the sample of {len(df)} married individuals, the proportion with any extramarital affair "
        f"was {prev_with_children:.3f} among those with children and {prev_without_children:.3f} among those without children."
    )
    explanation_parts.append(
        "I fit a logistic regression model predicting whether a respondent had any affair from an indicator for having children, "
        "controlling for age, years married, gender, religiousness, education, occupation, and self-rated marital satisfaction."
    )
    explanation_parts.append(
        f"In this model, the coefficient on having children was {children_coef:.3f} "
        f"(odds ratio {children_or:.3f}, p-value {children_p:.3g}), indicating {direction_text} odds of affairs for those with children."
    )

    if children_p >= 0.05:
        explanation_parts.append(
            "Because this effect is not statistically significant at the 5% level, the data do not provide strong evidence "
            "that having children meaningfully decreases extramarital affairs once other factors are accounted for."
        )
    else:
        if children_coef < 0:
            explanation_parts.append(
                "Because this negative association is statistically significant, the data support the conclusion that having children "
                "is associated with reduced odds of extramarital affairs, although the effect size should be interpreted in context."
            )
        else:
            explanation_parts.append(
                "Because this positive association is statistically significant, the data suggest that having children is actually "
                "associated with increased odds of extramarital affairs, contrary to the initial belief."
            )

    explanation_parts.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the claim that having children decreases extramarital affairs, "
        f"I assign a score of {response}, reflecting the direction, magnitude, and statistical significance of the estimated effect."
    )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

