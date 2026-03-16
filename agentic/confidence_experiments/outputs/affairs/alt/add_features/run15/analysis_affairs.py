import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for having any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children as a binary indicator: 1 = yes, 0 = no
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics by children status
    desc_group = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("has_affair", "mean"),
            n=("has_affair", "size"),
        )
        .reset_index()
    )

    # Logistic regression: probability of any affair ~ children + covariates
    # Use age, yearsmarried, religiousness, education, occupation, rating, gender.
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect of having children (children[T.yes] vs baseline no)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # In case of unexpected category coding, guard access.
    coef_key = None
    for key in params.index:
        if key.startswith("C(children)[T.") and "yes" in key:
            coef_key = key
            break

    if coef_key is None:
        # Fallback: try direct numeric children_yes effect
        logit_model2 = smf.logit(
            formula="has_affair ~ children_yes + age + yearsmarried + "
            "religiousness + education + occupation + rating + C(gender)",
            data=df,
        ).fit(disp=False)
        params = logit_model2.params
        pvalues = logit_model2.pvalues
        coef_key = "children_yes"
        model_used = "logit_children_yes"
    else:
        model_used = "logit_children_factor"

    coef = float(params[coef_key])
    pval = float(pvalues[coef_key])

    # Compute odds ratio for interpretability
    odds_ratio = float(np.exp(coef))

    # Also compute simple difference in proportions
    desc = desc_group.set_index("children")
    mean_affairs_yes = float(desc.loc["yes", "mean_affairs"])
    mean_affairs_no = float(desc.loc["no", "mean_affairs"])
    prop_any_yes = float(desc.loc["yes", "prop_any_affair"])
    prop_any_no = float(desc.loc["no", "prop_any_affair"])

    # Determine Likert-style response (0-100) and textual explanation.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"Descriptively, mean affair score is {mean_affairs_yes:.2f} for parents "
        f"and {mean_affairs_no:.2f} for non-parents."
    )
    explanation_lines.append(
        f"The proportion with any affair is {prop_any_yes:.3f} for parents "
        f"and {prop_any_no:.3f} for non-parents."
    )
    explanation_lines.append(
        f"Logistic regression ({model_used}) controlling for age, years married, "
        f"religiousness, education, occupation, rating, and gender yields an "
        f"effect of children on the log-odds of having any affair of {coef:.3f} "
        f"(odds ratio {odds_ratio:.3f}, p-value {pval:.4f})."
    )

    # Map effect + significance to Likert response.
    # Negative, significant coefficient -> children associated with fewer affairs.
    # Positive, significant coefficient -> associated with more affairs.
    # Non-significant -> little evidence of an association.
    alpha = 0.05
    if pval >= alpha:
        # No clear evidence of effect: respond "No" with low confidence.
        response = 30
        explanation_lines.append(
            "The effect of children is not statistically significant at the 5% level, "
            "so the data do not provide strong evidence that having children changes "
            "engagement in extramarital affairs after accounting for covariates."
        )
    else:
        # Significant effect: scale based on direction and magnitude.
        # Start from neutral 50 and move up/down depending on log-odds magnitude.
        # Cap at [0, 100].
        base = 50
        # Use a smooth scaling on the absolute log-odds, saturating around |coef|=1.5
        strength = min(abs(coef) / 1.5, 1.0)
        delta = int(round(40 * strength))
        if coef < 0:
            # Children decrease affairs -> Yes with higher score
            response = base + delta
            explanation_lines.append(
                "The coefficient for children is negative and statistically significant, "
                "indicating that, holding other factors constant, having children is "
                "associated with a lower likelihood of engaging in extramarital affairs."
            )
        else:
            # Children increase affairs -> Yes but in the opposite direction
            response = base + delta
            explanation_lines.append(
                "The coefficient for children is positive and statistically significant, "
                "indicating that, holding other factors constant, having children is "
                "associated with a higher likelihood of engaging in extramarital affairs."
            )

    explanation_lines.append(
        "The Likert-style score (0 = strong No, 100 = strong Yes) reflects both "
        "the statistical significance and the estimated strength of the relationship."
    )

    explanation = " ".join(explanation_lines)

    output = {"response": int(response), "explanation": explanation}

    # Write conclusion.txt exactly as required (JSON only).
    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

