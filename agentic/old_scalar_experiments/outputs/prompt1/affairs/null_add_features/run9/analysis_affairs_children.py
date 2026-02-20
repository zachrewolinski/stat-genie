import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create binary outcome: any extramarital affair in last year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-wise summaries by children
    group_summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
    )

    print("Group-wise summaries by children:")
    print(group_summary)
    print()

    # Logistic regression controlling for standard covariates
    # Use a reduced set of predictors focusing on classic Fair model variables.
    formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression results (has_affair as outcome):")
    print(logit_model.summary())

    # Print odds ratio for having children vs not (if coded that way)
    params = logit_model.params
    conf_int = logit_model.conf_int()
    or_table = params.to_frame("coef")
    or_table["odds_ratio"] = or_table["coef"].apply(lambda x: float(np.exp(x)))
    or_table["ci_lower"] = conf_int[0].apply(lambda x: float(np.exp(x)))
    or_table["ci_upper"] = conf_int[1].apply(lambda x: float(np.exp(x)))

    print("\nOdds ratios (exponentiated coefficients):")
    print(or_table.loc[[i for i in or_table.index if "children" in i]])


if __name__ == "__main__":
    main()
