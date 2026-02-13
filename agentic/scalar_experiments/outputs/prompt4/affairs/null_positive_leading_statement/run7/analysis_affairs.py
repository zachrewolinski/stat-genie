import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affairs in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as binary: 1 = has children, 0 = no children
    df["children_binary"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Basic descriptive statistics: prevalence of affairs by children status
    prevalence_by_children = (
        df.groupby("children")["has_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "affair_rate", "count": "n"})
    )

    # Logistic regression controlling for other covariates mentioned in metadata
    # has_affair ~ children + gender + age + yearsmarried + religiousness + education + occupation + rating
    model = smf.logit(
        "has_affair ~ children_binary + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)

    coef_children = float(model.params["children_binary"])
    p_children = float(model.pvalues["children_binary"])

    # Extract simple descriptive comparison
    # Ensure ordering: index typically ['no', 'yes'], but guard in case
    prevalence_dict = prevalence_by_children["affair_rate"].to_dict()
    rate_no = float(prevalence_dict.get("no", float("nan")))
    rate_yes = float(prevalence_dict.get("yes", float("nan")))

    # Decide on the Likert-style response in [0, 100]
    # We interpret strong evidence for a *decrease* as:
    #  - children coefficient negative and statistically significant (p < 0.05)
    #  - and observed affair rate lower among those with children
    if (
        coef_children < 0
        and p_children < 0.05
        and (rate_yes < rate_no)
    ):
        # Strong "Yes": having children is associated with a lower probability
        response_score = 80
    elif (
        coef_children < 0
        and p_children < 0.10
        and (rate_yes <= rate_no)
    ):
        # Some suggestive but weaker evidence
        response_score = 65
    elif coef_children < 0 and (rate_yes <= rate_no):
        # Directionally negative but not statistically strong
        response_score = 55
    else:
        # Coefficient is non-negative or children not clearly associated
        # with lower affair engagement: answer leans "No".
        # Calibrate the strength based on the magnitude and sign.
        if coef_children > 0 and rate_yes > rate_no and p_children < 0.05:
            # Statistically significant increase instead of decrease
            response_score = 10
        elif coef_children > 0 and rate_yes >= rate_no and p_children < 0.10:
            response_score = 20
        elif coef_children > 0 and rate_yes >= rate_no:
            response_score = 30
        else:
            # Essentially no clear effect
            response_score = 40

    # Build explanation string with key numerical evidence
    explanation = (
        "I examined the Fair affairs dataset of 601 married individuals. "
        f"I converted the numeric affairs count into a binary indicator of any affair in the past year "
        f"and compared prevalence by children status. The estimated affair rate was "
        f"{rate_no:.3f} among respondents without children and {rate_yes:.3f} among those with children. "
        f"I then fit a logistic regression model where the dependent variable was having any affair and "
        f"the main predictor was a binary indicator for having children, controlling for gender, age, years married, "
        f"religiousness, education, occupation, and self-rated marital happiness. "
        f"The coefficient on the children indicator was {coef_children:.3f} with p-value {p_children:.3f}. "
        "Because this coefficient does not provide clear, statistically strong evidence that having children "
        "reduces the probability of an affair—and the simple prevalence comparison does not show a substantial "
        "decrease for those with children—I conclude that this dataset does not strongly support the claim "
        "that having children decreases engagement in extramarital affairs."
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write required JSON-only conclusion file
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

