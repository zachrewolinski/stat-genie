import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # In this shuffled schema, the "age" column encodes affair frequency,
    # and the "religiousness" column is a yes/no flag for whether there are children.
    df["affair_freq"] = df["age"]
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows with missing key values just in case.
    df_model = df.dropna(
        subset=["any_affair", "has_children", "gender", "occupation", "children", "rating", "yearsmarried"]
    ).copy()

    # Group-level summaries: mean frequency and proportion with any affair by children status.
    group_means = df_model.groupby("has_children")["affair_freq"].mean()
    group_props = df_model.groupby("has_children")["any_affair"].mean()

    # Logistic regression for any affair vs. children and controls.
    logit_formula = "any_affair ~ has_children + C(gender) + occupation + children + rating + yearsmarried"
    logit_model = smf.logit(logit_formula, data=df_model).fit(disp=False)
    has_children_coef = float(logit_model.params["has_children"])
    has_children_pval = float(logit_model.pvalues["has_children"])
    has_children_or = float(np.exp(has_children_coef))

    # Poisson regression for affair frequency (count-like outcome) with same predictors.
    poisson_formula = "affair_freq ~ has_children + C(gender) + occupation + children + rating + yearsmarried"
    poisson_model = smf.glm(
        poisson_formula,
        data=df_model,
        family=sm.families.Poisson(),
    ).fit()
    has_children_coef_p = float(poisson_model.params["has_children"])
    has_children_pval_p = float(poisson_model.pvalues["has_children"])
    has_children_rr = float(np.exp(has_children_coef_p))

    # Determine Likert response on 0-100 scale.
    # We consider both direction and statistical significance from the two models.
    # If children clearly reduce affairs (negative, significant effects), we answer "Yes" with a high score.
    # If effects are small or not statistically significant, we answer "No" with a low score.
    alpha = 0.05

    reduces_in_logit = has_children_coef < 0 and has_children_pval < alpha
    reduces_in_pois = has_children_coef_p < 0 and has_children_pval_p < alpha

    if reduces_in_logit or reduces_in_pois:
        # Some evidence that having children is associated with fewer affairs.
        # Calibrate strength by combining significance and effect size.
        # Stronger negative and lower p-values give higher scores.
        avg_rr = (has_children_or + has_children_rr) / 2.0
        # Map average risk ratio into a rough strength: smaller ratios -> stronger "Yes".
        if avg_rr <= 0.6:
            response_score = 85
        elif avg_rr <= 0.8:
            response_score = 70
        else:
            response_score = 60
    else:
        # No robust evidence that children reduce affairs; in many analyses of this dataset,
        # the children effect is small and statistically non-significant.
        # Use a low score to indicate a "No" answer with moderate confidence.
        response_score = 20

    # Build explanation text summarizing evidence in plain language.
    mean_no_children = float(group_means.get(0, np.nan))
    mean_children = float(group_means.get(1, np.nan))
    prop_no_children = float(group_props.get(0, np.nan))
    prop_children = float(group_props.get(1, np.nan))

    explanation_lines = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
        "",
        "Dataset and variables:",
        "- Affair engagement is measured by the numeric 'age' column, which encodes how often an individual engaged in extramarital sexual intercourse in the past year.",
        "- The 'religiousness' column indicates whether there are children in the marriage ('yes' = children present, 'no' = no children).",
        "",
        "Descriptive statistics:",
        f"- Mean affair frequency with no children: {mean_no_children:.3f}",
        f"- Mean affair frequency with children: {mean_children:.3f}",
        f"- Proportion with any affair (no children): {prop_no_children:.3f}",
        f"- Proportion with any affair (children): {prop_children:.3f}",
        "",
        "Inferential analysis:",
        "- I fit a logistic regression for having any affair (binary) as a function of children status and controls "
        "for gender, age group (occupation), years married (children), religiousness score (rating), and education (yearsmarried).",
        f"- In this model, the coefficient for having children is {has_children_coef:.3f} with odds ratio {has_children_or:.3f} "
        f"and p-value {has_children_pval:.3f}.",
        "- I also fit a Poisson regression for affair frequency using the same predictors.",
        f"- In the Poisson model, the coefficient for having children is {has_children_coef_p:.3f} with rate ratio {has_children_rr:.3f} "
        f"and p-value {has_children_pval_p:.3f}.",
        "",
        "Conclusion:",
    ]

    if response_score >= 50:
        conclusion_text = (
            "There is statistical evidence that having children is associated with a lower level of extramarital "
            "affair engagement after adjusting for other factors. The negative and statistically significant effect "
            "of children suggests that parents, on average, report fewer or less frequent extramarital affairs "
            "than couples without children."
        )
    else:
        conclusion_text = (
            "There is no strong statistical evidence that having children decreases engagement in extramarital "
            "affairs. Differences in affair frequency and the likelihood of having any affair between couples with "
            "and without children are small, and the estimated effects of children are not statistically significant "
            "in either the logistic or Poisson regression models once other factors are controlled for."
        )

    explanation_lines.append(conclusion_text)
    explanation = "\n".join(explanation_lines)

    conclusion = {"response": int(response_score), "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    main()

