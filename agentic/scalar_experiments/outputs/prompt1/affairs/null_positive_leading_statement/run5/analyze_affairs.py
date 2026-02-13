import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create binary indicators
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_binary"] = (df["children"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics
    grouped = df.groupby("children")["affairs"]
    desc = grouped.agg(["mean", "std", "median", "count"])

    any_affair_rate = df.groupby("children")["any_affair"].mean()

    print("Affairs count by children status:")
    print(desc)
    print()
    print("Proportion with any affair by children status:")
    print(any_affair_rate)
    print()

    # Logistic regression on any affair, including other covariates as controls
    # Treat children as a factor to get an interpretable coefficient
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness"
        " + education + occupation + rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression results (any_affair as outcome):")
    print(logit_model.summary())
    print()

    # Extract effect of children (yes vs no)
    params = logit_model.params
    pvalues = logit_model.pvalues

    # statsmodels encodes as C(children)[T.yes] with "no" as baseline
    child_term = "C(children)[T.yes]"
    if child_term in params:
        coef = params[child_term]
        pval = pvalues[child_term]
        odds_ratio = float(np.exp(coef))
    else:
        coef = np.nan
        pval = np.nan
        odds_ratio = np.nan

    print(f"Children coefficient (yes vs no): {coef:.4f}")
    print(f"P-value: {pval:.4g}")
    print(f"Odds ratio: {odds_ratio:.4f}")


if __name__ == "__main__":
    main()

