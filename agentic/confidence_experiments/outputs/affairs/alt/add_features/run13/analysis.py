import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Children indicator: 1 = yes, 0 = no
    df["children_yes"] = df["children"].str.lower().eq("yes").astype(int)

    # Simple descriptive stats
    group = df.groupby("children_yes")["any_affair"]
    rate_with_children = group.mean().get(1, np.nan)
    rate_without_children = group.mean().get(0, np.nan)
    n_with_children = group.count().get(1, 0)
    n_without_children = group.count().get(0, 0)

    print("Sample sizes:")
    print(f"  With children:    n = {n_with_children}")
    print(f"  Without children: n = {n_without_children}")
    print()
    print("Proportion with any extramarital affair:")
    print(f"  With children:    {rate_with_children:.3f}")
    print(f"  Without children: {rate_without_children:.3f}")
    print()

    # Logistic regression: probability of any affair ~ children only
    y = df["any_affair"]
    X1 = sm.add_constant(df[["children_yes"]])
    logit_children_only = sm.Logit(y, X1, missing="drop").fit(disp=False)

    coef1 = logit_children_only.params["children_yes"]
    pval1 = logit_children_only.pvalues["children_yes"]
    or1 = float(np.exp(coef1))

    print("Logistic regression (any_affair ~ children):")
    print(f"  coef(children_yes) = {coef1:.3f}")
    print(f"  odds ratio          = {or1:.3f}")
    print(f"  p-value             = {pval1:.4f}")
    print()

    # Logistic regression with standard controls from the classic Fair dataset
    controls = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    available_controls = [c for c in controls if c in df.columns]
    X2 = sm.add_constant(df[["children_yes"] + available_controls])

    logit_with_controls = sm.Logit(y, X2, missing="drop").fit(disp=False)
    coef2 = logit_with_controls.params["children_yes"]
    pval2 = logit_with_controls.pvalues["children_yes"]
    or2 = float(np.exp(coef2))

    print("Logistic regression with controls (any_affair ~ children + covariates):")
    print(f"  controls: {available_controls}")
    print(f"  coef(children_yes) = {coef2:.3f}")
    print(f"  odds ratio          = {or2:.3f}")
    print(f"  p-value             = {pval2:.4f}")


if __name__ == "__main__":
    main()

