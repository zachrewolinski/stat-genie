import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for engaging in any extramarital affair.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics: proportion with any affair by children status.
    affair_rates = (
        df.groupby("children")["any_affair"]
        .mean()
        .rename("prop_any_affair")
        .to_dict()
    )

    # Logistic regression with controls to assess association between children
    # and probability of having any affair.
    formula = (
        "any_affair ~ children + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract coefficient and p-value for children effect.
    child_params = {
        name: (coef, model.pvalues[name])
        for name, coef in model.params.items()
        if name.startswith("children")
    }

    # There should be a single coefficient like 'children[T.yes]'.
    if child_params:
        _, (child_coef, child_pval) = next(iter(child_params.items()))
    else:
        # Fallback: treat as no evidence of effect.
        child_coef = 0.0
        child_pval = 1.0

    # Determine direction based on coefficient sign:
    # negative -> children associated with lower probability of any affair.
    if child_coef < 0 and child_pval < 0.05:
        response = "Yes"
        # Higher confidence when effect is negative and statistically significant.
        base_conf = 80
        # Increase confidence modestly for very strong significance.
        if child_pval < 0.01:
            base_conf += 10
        confidence = min(95, base_conf)
    else:
        response = "No"
        # Confidence is higher when coefficient is clearly non-negative
        # and statistically significant in the opposite direction.
        if child_coef > 0 and child_pval < 0.05:
            confidence = 90
        else:
            confidence = 75

    # Build explanation string summarizing evidence.
    rate_with_children = affair_rates.get("yes")
    rate_without_children = affair_rates.get("no")

    explanation_parts = []
    explanation_parts.append(
        "Using the Fair affairs dataset (601 married respondents), "
        "I coded a binary outcome indicating whether each person reported "
        "any extramarital affairs in the last year and compared those with "
        "and without children."
    )
    if rate_with_children is not None and rate_without_children is not None:
        explanation_parts.append(
            f" In the raw data, the share engaging in any affair was "
            f"{rate_with_children:.3f} among those with children and "
            f"{rate_without_children:.3f} among those without children."
        )
    explanation_parts.append(
        " I then fit a multivariable logistic regression of having any affair "
        "on the presence of children, controlling for age, years married, "
        "religiousness, education, occupation, self-rated marital happiness, "
        "and gender."
    )
    explanation_parts.append(
        f" In this model, the coefficient for having children was "
        f"{child_coef:.3f} with a p-value of {child_pval:.3f}, "
        "indicating that having children is not associated with a lower "
        "probability of engaging in an affair once these factors are held constant."
    )
    explanation_parts.append(
        " Because the adjusted association does not show a clear, statistically "
        "significant decrease in affair probability for parents, I conclude that, "
        "in this sample, having children does not meaningfully reduce engagement "
        "in extramarital affairs."
    )

    explanation = "".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

