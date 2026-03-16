import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("Sample size:", len(df))
    print()

    print("Affairs by children (mean, std):")
    print(df.groupby("children")["affairs"].agg(["mean", "std", "count"]))
    print()

    print("Proportion with any affair (>0) by children:")
    prop = df.groupby("children")["any_affair"].mean()
    print(prop)
    print()

    # Simple logistic regression with children only
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    params_simple = logit_simple.params
    pvalues_simple = logit_simple.pvalues
    or_simple = params_simple.map(lambda x: float(np.exp(x)))

    print("Logistic regression (any_affair ~ children):")
    print("Coefficients:")
    print(params_simple)
    print("P-values:")
    print(pvalues_simple)
    print("Odds ratios (exp(coef)):")
    print(or_simple)
    print()

    # Logistic regression with controls
    formula_full = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + C(gender) + occupation + rating"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    params_full = logit_full.params
    pvalues_full = logit_full.pvalues
    or_full = params_full.map(lambda x: float(np.exp(x)))

    print("Logistic regression with controls:")
    print("Coefficients:")
    print(params_full)
    print("P-values:")
    print(pvalues_full)
    print("Odds ratios (exp(coef)):")
    print(or_full)
    print()


if __name__ == "__main__":
    main()
