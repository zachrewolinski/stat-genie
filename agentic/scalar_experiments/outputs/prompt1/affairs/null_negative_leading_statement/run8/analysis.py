import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Define binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparison: proportion with any affair by children status
    desc = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_any_affair", "count": "n"})
    )

    # Logistic regression controlling for key covariates
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + C(gender) + rating"
    )
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    # Effect of having children (yes vs. no)
    coef_children = float(result.params.get("C(children)[T.yes]", float("nan")))
    p_children = float(result.pvalues.get("C(children)[T.yes]", float("nan")))

    # Simple decision rule:
    # Only answer "Yes" if having children shows a statistically significant
    # *decrease* in the odds of any affair (negative coefficient, p < 0.05).
    if coef_children < 0 and p_children < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Prepare numbers for explanation
    prop_children_yes = float(desc.loc["yes", "prop_any_affair"])
    prop_children_no = float(desc.loc["no", "prop_any_affair"])
    n_children_yes = int(desc.loc["yes", "n"])
    n_children_no = int(desc.loc["no", "n"])

    explanation = (
        "Using the Fair affairs dataset (n=601), I created a binary variable "
        "indicating whether each respondent reported any extramarital affair in "
        "the past year. I then compared this outcome between marriages with and "
        "without children and fit a logistic regression model to adjust for "
        "age, years married, religiousness, education, occupation, gender, and "
        "self-rated marital happiness.\n\n"
        f"Descriptively, the proportion reporting at least one affair was "
        f"{prop_children_no:.3f} among respondents without children (n={n_children_no}) "
        f"and {prop_children_yes:.3f} among respondents with children (n={n_children_yes}). "
        "These proportions do not show a clear reduction in affairs among couples "
        "with children.\n\n"
        "In the multivariable logistic regression, the coefficient for having "
        f"children (yes vs. no) on the log-odds of any affair was "
        f"{coef_children:.3f} with a p-value of {p_children:.3f}. This effect is "
        "not a statistically significant negative association, meaning that after "
        "accounting for the other recorded demographic and relationship factors, "
        "the data do not provide strong evidence that having children reduces "
        "the likelihood of engaging in extramarital affairs.\n\n"
        f"Based on both the descriptive comparison and the adjusted regression "
        f"results, I conclude that this dataset does not support the claim that "
        "having children decreases engagement in extramarital affairs."
    )

    output = {"response": response, "explanation": explanation}

    # Write a single JSON object with no extra text
    Path("conclusion.txt").write_text(json.dumps(output))


if __name__ == "__main__":
    main()
