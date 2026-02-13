import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive stats by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_with_affair=("has_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))
    print()

    # Linear model for affair count (treating affairs as numeric score)
    # Using log(affairs + 1) to stabilize variance
    df["log_affairs_plus1"] = np.log(df["affairs"] + 1)

    formula_count = (
        "log_affairs_plus1 ~ children_yes + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )

    lm = smf.ols(formula=formula_count, data=df).fit()
    print("OLS model on log(affairs + 1):")
    print(lm.summary().tables[1])
    print()

    # Logistic model for any affair vs none
    formula_logit = (
        "has_affair ~ children_yes + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    logit = smf.logit(formula=formula_logit, data=df).fit(disp=False)
    print("Logistic regression for having any affair:")
    print(logit.summary().tables[1])


if __name__ == "__main__":
    main()
