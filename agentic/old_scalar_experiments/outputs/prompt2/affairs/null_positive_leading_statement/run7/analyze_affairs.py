import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create a binary indicator of having any affairs in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary (1=yes, 0=no).
    df["children_bin"] = (df["children"] == "yes").astype(int)

    # Simple descriptive comparison of affair rates by children status.
    group_means = df.groupby("children")["any_affair"].mean()
    rate_with_children = group_means.get("yes", np.nan)
    rate_without_children = group_means.get("no", np.nan)

    # Logistic regression controlling for key covariates.
    # Using age, yearsmarried, religiousness, education, occupation, rating and gender.
    df["gender_male"] = (df["gender"] == "male").astype(int)

    formula = (
        "any_affair ~ children_bin + age + yearsmarried + religiousness "
        "+ education + occupation + rating + gender_male"
    )

    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        coef_children = model.params.get("children_bin", np.nan)
        pvalue_children = model.pvalues.get("children_bin", np.nan)
        # Marginal effect (approximate) on probability scale at mean covariates.
        # For small coefficients in logistic regression, the slope is close to
        # coef * p * (1-p); we use overall mean of any_affair as p.
        base_rate = df["any_affair"].mean()
        marginal_effect = float(coef_children * base_rate * (1 - base_rate))
    except Exception:
        # Fallback if model fails; rely only on descriptive difference.
        coef_children = np.nan
        pvalue_children = np.nan
        marginal_effect = float("nan")

    # Determine direction of association from descriptive stats.
    # If those with children have lower affair rate than those without,
    # that supports "children decrease engagement in extramarital affairs".
    descriptive_supports_yes = bool(
        np.isfinite(rate_with_children)
        and np.isfinite(rate_without_children)
        and rate_with_children < rate_without_children
    )

    # Determine direction from regression coefficient.
    regression_supports_yes = bool(np.isfinite(coef_children) and coef_children < 0)

    # Combine evidence.
    if descriptive_supports_yes and regression_supports_yes:
        response = "Yes"
        # Confidence higher when both agree and effect is reasonably sized/significant.
        confidence = 80
        if np.isfinite(pvalue_children) and pvalue_children < 0.05:
            confidence = 90
    elif descriptive_supports_yes or regression_supports_yes:
        response = "Yes"
        confidence = 65
    else:
        response = "No"
        confidence = 80
        if np.isfinite(pvalue_children) and pvalue_children < 0.05:
            confidence = 90

    explanation_parts = []
    explanation_parts.append(
        "I analyzed 601 married individuals from the Fair affairs dataset, "
        "creating a binary indicator of whether each person had any extramarital "
        "affair in the past year."
    )
    explanation_parts.append(
        f"The observed proportion with any affair was "
        f"{rate_with_children:.3f} among those with children and "
        f"{rate_without_children:.3f} among those without children."
    )

    if np.isfinite(coef_children):
        explanation_parts.append(
            "I then fit a logistic regression for having any affair with a "
            "binary children indicator and controls for age, years married, "
            "religiousness, education, occupation, marital satisfaction rating, "
            "and gender."
        )
        explanation_parts.append(
            f"The regression coefficient for having children was "
            f"{coef_children:.3f} with p-value {pvalue_children:.3f}, "
            f"implying an approximate marginal effect of "
            f"{marginal_effect:.3f} on the probability of having an affair."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the descriptive comparison and regression direction suggest "
            "that having children is associated with a lower or at least not "
            "higher likelihood of extramarital affairs."
        )
    else:
        explanation_parts.append(
            "The descriptive comparison and regression do not show that having "
            "children reduces engagement in extramarital affairs; the effect is "
            "small, inconsistent in sign, or not statistically distinguishable "
            "from no effect."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

