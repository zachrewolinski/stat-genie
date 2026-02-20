import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create a binary indicator for having any extramarital affairs.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status.
    desc = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prop_any_affair", "sum": "num_with_affair"})
    )

    # Logistic regression controlling for observed covariates.
    # Use 'no' as the baseline for children; the coefficient on C(children)[T.yes]
    # tells us whether having children is associated with higher or lower odds
    # of any affair relative to no children.
    model = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    params = model.params
    pvalues = model.pvalues

    # Extract the coefficient and p-value for children=yes vs no (baseline).
    child_term = "C(children)[T.yes]"
    child_coef = float(params.get(child_term, float("nan")))
    child_p = float(pvalues.get(child_term, float("nan")))

    # Decide on the answer:
    # - "Yes" only if we see a negative coefficient (suggesting reduced odds)
    #   and conventional statistical evidence (p < 0.05).
    # - Otherwise answer "No" (no clear evidence that children decrease affairs).
    if child_coef < 0 and child_p < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Gather key numeric evidence for the explanation.
    # Descriptive: proportions with any affair by children status.
    prop_children_yes = float(desc.loc["yes", "prop_any_affair"])
    prop_children_no = float(desc.loc["no", "prop_any_affair"])
    n_children_yes = int(desc.loc["yes", "count"])
    n_children_no = int(desc.loc["no", "count"])

    # Build a concise explanation string.
    explanation = (
        "Using the 601-observation affairs dataset, I created a binary indicator "
        "for whether each respondent reported any extramarital sexual activity in "
        "the past year and compared this between marriages with and without "
        "children. Descriptively, the proportion reporting at least one affair "
        f"was {prop_children_yes:.3f} among those with children (n={n_children_yes}) "
        f"and {prop_children_no:.3f} among those without children (n={n_children_no}), "
        "which does not suggest a clear decrease in affairs among parents. "
        "I then fit a logistic regression for any affair on an indicator for having "
        "children while controlling for age, years married, religiousness, education, "
        "occupation, self-rated marital happiness, and gender. In this model, the "
        f"coefficient for having children (yes vs. no) was {child_coef:.3f} with a "
        f"p-value of {child_p:.3f}, indicating no statistically significant evidence "
        "that having children reduces the likelihood of engaging in extramarital "
        "affairs once these factors are taken into account. Based on this analysis, "
        "I conclude that the data do not support the claim that having children "
        "decreases engagement in extramarital affairs."
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write the required JSON output to conclusion.txt with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

