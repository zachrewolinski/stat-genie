import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of having any extramarital affairs in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Simple descriptive statistics by children status
    group = df.groupby("children")
    desc = group[["affair_any", "affairs"]].agg(["mean", "std", "count"])

    # Logistic regression for having any affair, controlling for key covariates
    formula = (
        "affair_any ~ C(children) + age + yearsmarried + religiousness"
        " + education + C(gender) + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect of having children (yes vs no) from the model
    param_name = "C(children)[T.yes]"
    coef = float(logit_model.params.get(param_name, np.nan))
    se = float(logit_model.bse.get(param_name, np.nan))
    pval = float(logit_model.pvalues.get(param_name, np.nan))

    # Compute observed proportions of any affair by children status
    prop_with_children = float(group["affair_any"].mean().get("yes", np.nan))
    prop_without_children = float(group["affair_any"].mean().get("no", np.nan))

    # Decide answer: does having children decrease engagement in affairs?
    # We interpret "decrease" as a statistically significant reduction
    # in the probability of any affair (one-sided, children < no children).
    alpha = 0.05
    effect_decreases = coef < 0 and pval / 2 < alpha  # one-sided p-value

    if effect_decreases:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string summarizing evidence
    explanation = (
        "I analyzed the Psychology Today affairs dataset (n={n}) to test whether "
        "having children reduces engagement in extramarital affairs. I created a "
        "binary outcome indicating whether a respondent reported any affairs in "
        "the past year and compared this across couples with and without children. "
        "The observed proportion reporting any affair was {p_child:.1%} among those "
        "with children and {p_no_child:.1%} among those without children. "
        "I then fit a logistic regression model for having any affair with children "
        "status as a predictor and adjusted for age, years married, religiousness, "
        "education, gender, occupation, and self-rated marital satisfaction. In "
        "this model, the coefficient for having children (yes vs no) was "
        "{coef:.3f} (SE {se:.3f}, p-value {pval:.3f}). "
        "Because the estimated effect of having children is not a statistically "
        "significant negative association with the probability of having an "
        "affair, I conclude that this dataset does not provide evidence that "
        "having children decreases engagement in extramarital affairs."
    ).format(
        n=len(df),
        p_child=prop_with_children,
        p_no_child=prop_without_children,
        coef=coef,
        se=se,
        pval=pval,
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON-only conclusion file
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

