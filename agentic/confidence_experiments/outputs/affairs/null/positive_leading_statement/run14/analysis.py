import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["child_yes"] = (df["children"] == "yes").astype(int)

    # Basic group summaries
    group = df.groupby("children")
    summary = group["affairs"].agg(["mean", "median"])
    prop_any = group["has_affair"].mean()

    print("=== Descriptive statistics by children ===")
    print(summary)
    print("\nProportion with any affair (affairs > 0):")
    print(prop_any)

    # Logistic regression of having any affair on children (unadjusted)
    logit_simple = smf.logit("has_affair ~ child_yes", data=df).fit(disp=False)
    print("\n=== Logistic regression: has_affair ~ child_yes ===")
    print(logit_simple.summary())
    coef = logit_simple.params["child_yes"]
    pval = logit_simple.pvalues["child_yes"]
    odds_ratio = float(np.exp(coef))
    print(f"\nchild_yes coef: {coef:.4f}, p-value: {pval:.4g}, odds ratio: {odds_ratio:.3f}")

    # Logistic regression with controls
    formula_full = (
        "has_affair ~ child_yes + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("\n=== Logistic regression with controls ===")
    print(logit_full.summary())
    coef_full = logit_full.params["child_yes"]
    pval_full = logit_full.pvalues["child_yes"]
    odds_ratio_full = float(np.exp(coef_full))
    print(
        f"\n(child_yes, full model) coef: {coef_full:.4f}, "
        f"p-value: {pval_full:.4g}, odds ratio: {odds_ratio_full:.3f}"
    )


if __name__ == "__main__":
    main()

