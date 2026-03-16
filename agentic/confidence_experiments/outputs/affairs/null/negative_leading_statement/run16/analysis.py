import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives by children status
    grouped = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            mean_any_affair=("any_affair", "mean"),
            mean_affairs=("affairs", "mean"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(grouped.to_string(index=False))
    print()

    # Unadjusted logistic regression: any_affair ~ children
    print("Unadjusted logistic regression (any_affair ~ C(children)):")
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print(model_unadj.summary2())
    print()

    # Adjusted logistic regression with key covariates
    formula_adj = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    print("Adjusted logistic regression with covariates:")
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print(model_adj.summary2())


if __name__ == "__main__":
    main()

