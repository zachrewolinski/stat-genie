import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity within this script
    df = df.rename(
        columns={
            "feature2": "affair_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Binary indicator: any extramarital affair in past year
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Basic group statistics by children status
    group = df.groupby("children", observed=True)
    mean_affair = group["affair_freq"].mean()
    prop_any = group["any_affair"].mean()
    counts = group["any_affair"].count()

    print("=== Descriptive statistics by children status ===")
    for child_status in mean_affair.index:
        print(
            f"children = {child_status:3s} | n = {counts[child_status]:3d} | "
            f"mean affair freq = {mean_affair[child_status]:.3f} | "
            f"proportion any affair = {prop_any[child_status]:.3f}"
        )

    # Simple logistic regression: any_affair ~ children
    # children is treated as a categorical variable with baseline determined by statsmodels
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("\n=== Logistic regression: any_affair ~ C(children) ===")
    print(logit_simple.summary())

    # Slightly richer model including standard covariates
    formula_full = (
        "any_affair ~ C(children) + C(gender) + age + years_married + "
        "religiousness + education + occupation + marriage_rating"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("\n=== Logistic regression with controls ===")
    print(logit_full.summary())

    # Extract key coefficient and statistics for children effect from full model
    # We expect a term named like C(children)[T.yes] or C(children)[T.no]
    params = logit_full.params
    conf_int = logit_full.conf_int()
    pvalues = logit_full.pvalues

    child_term = None
    for name in params.index:
        if name.startswith("C(children)[T."):
            child_term = name
            break

    if child_term is None:
        print("\n[WARN] Could not locate children coefficient in full model.")
        effect_summary = None
    else:
        coef = params[child_term]
        pval = pvalues[child_term]
        ci_low, ci_high = conf_int.loc[child_term]
        odds_ratio = float(np.exp(coef))
        effect_summary = {
            "term": child_term,
            "coef": float(coef),
            "odds_ratio": odds_ratio,
            "p_value": float(pval),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }

        print("\n=== Children effect in full model ===")
        print(json.dumps(effect_summary, indent=2))


if __name__ == "__main__":
    main()

