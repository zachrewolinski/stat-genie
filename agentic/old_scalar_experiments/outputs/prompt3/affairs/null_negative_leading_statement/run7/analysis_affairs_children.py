import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for whether the respondent had any affairs in the last year.
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by presence of children.
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any=("affair_any", "mean"),
            count=("affair_any", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))
    print()

    # Logistic regression for any affair ~ children only.
    model1 = smf.logit("affair_any ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression: affair_any ~ C(children)")
    print(model1.summary())
    print()

    # Logistic regression controlling for key covariates from the metadata.
    formula_full = (
        "affair_any ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model2 = smf.logit(formula_full, data=df).fit(disp=False)
    print("Logistic regression with controls:")
    print(formula_full)
    print(model2.summary())
    print()

    # Extract and print the children coefficient from the controlled model for clarity.
    coef_children = model2.params.get("C(children)[T.yes]", float("nan"))
    pval_children = model2.pvalues.get("C(children)[T.yes]", float("nan"))
    print("Effect of having children (controlled model):")
    print(f"  Coefficient for C(children)[T.yes]: {coef_children:.4f}")
    print(f"  p-value: {pval_children:.4g}")


if __name__ == "__main__":
    main()

