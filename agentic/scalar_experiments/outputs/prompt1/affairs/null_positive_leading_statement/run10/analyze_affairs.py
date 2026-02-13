import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    prop_affair = group["has_affair"].mean()
    counts = group.size()

    # Logistic regression of any affair on children, controlling for covariates
    # Children is treated as a categorical yes/no factor.
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Effect of having children relative to not having children
    coef_children = float(logit_model.params.get("C(children)[T.yes]", np.nan))
    pval_children = float(logit_model.pvalues.get("C(children)[T.yes]", np.nan))

    # Prepare numeric summaries
    mean_affairs_yes = float(mean_affairs.get("yes", np.nan))
    mean_affairs_no = float(mean_affairs.get("no", np.nan))
    prop_affair_yes = float(prop_affair.get("yes", np.nan))
    prop_affair_no = float(prop_affair.get("no", np.nan))
    n_yes = int(counts.get("yes", 0))
    n_no = int(counts.get("no", 0))

    # Decide answer: does having children decrease engagement in extramarital affairs?
    # We require that both descriptive statistics and the regression coefficient
    # point in the same direction, with the regression effect statistically
    # distinguishable from zero at the 5% level.
    response: str
    if (
        not np.isnan(coef_children)
        and pval_children < 0.05
        and mean_affairs_yes < mean_affairs_no
        and prop_affair_yes < prop_affair_no
        and coef_children < 0
    ):
        response = "Yes"
        interpretation = (
            "Having children is associated with lower engagement in extramarital affairs."
        )
    else:
        response = "No"
        interpretation = (
            "The data do not provide clear evidence that having children decreases "
            "engagement in extramarital affairs."
        )

    explanation = (
        f"The dataset contains {n_no} individuals without children and {n_yes} with children. "
        f"The mean affair score is {mean_affairs_no:.2f} for those without children and "
        f"{mean_affairs_yes:.2f} for those with children. The proportion reporting at least one "
        f"affair is {prop_affair_no:.2%} without children versus {prop_affair_yes:.2%} with children. "
        f"A logistic regression of any affair on children, controlling for age, years married, "
        f"religiousness, education, occupation, marital rating, and gender, yields a coefficient "
        f"of {coef_children:.2f} (p = {pval_children:.3f}) for having children relative to not "
        f"having children. {interpretation}"
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

