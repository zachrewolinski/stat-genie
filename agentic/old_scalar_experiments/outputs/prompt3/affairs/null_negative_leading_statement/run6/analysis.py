import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic summaries by children status
    group_any = df.groupby("children")["affair_any"].agg(["mean", "sum", "count"])
    group_freq = df.groupby("children")["affairs"].agg(["mean", "median"])

    print("=== Descriptive statistics by children status ===")
    print(group_any)
    print()
    print("Affair frequency (all respondents):")
    print(group_freq)
    print()

    # Two-proportion z-test for any affair vs none
    # Ensure deterministic ordering by index label
    if set(group_any.index) != {"yes", "no"}:
        raise ValueError(f"Unexpected children categories: {group_any.index.tolist()}")

    counts = group_any.loc[["no", "yes"], "sum"].to_numpy()
    nobs = group_any.loc[["no", "yes"], "count"].to_numpy()
    stat, pval = proportions_ztest(count=counts, nobs=nobs)

    prop_no = group_any.loc["no", "mean"]
    prop_yes = group_any.loc["yes", "mean"]

    print("=== Two-proportion z-test: any affair (children=no vs yes) ===")
    print(f"Proportion with affair, no children:  {prop_no:.3f}")
    print(f"Proportion with affair, children:     {prop_yes:.3f}")
    print(f"Difference (children - no children):  {prop_yes - prop_no:.3f}")
    print(f"z statistic: {stat:.3f}, p-value: {pval:.4f}")
    print()

    # Logistic regression controlling for demographic and relationship factors
    formula = (
        "affair_any ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(occupation) + C(gender) + rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("=== Logistic regression: predictors of any affair ===")
    print(logit_model.summary())
    print()

    # Extract effect of having children (yes vs no)
    child_term = "C(children)[T.yes]"
    if child_term in logit_model.params:
        coef = logit_model.params[child_term]
        p_child = logit_model.pvalues[child_term]
        odds_ratio = float(np.exp(coef))
        print("Effect of children (yes vs no):")
        print(f"  Log-odds coefficient: {coef:.3f}")
        print(f"  Odds ratio:           {odds_ratio:.3f}")
        print(f"  p-value:              {p_child:.4f}")
    else:
        print("Children term not found in model parameters.")


if __name__ == "__main__":
    main()

