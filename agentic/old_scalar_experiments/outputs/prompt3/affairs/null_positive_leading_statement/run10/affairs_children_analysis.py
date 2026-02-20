import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))
    print()

    # Logistic regression of any_affair on children and covariates
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression results (any_affair as outcome):")
    print(logit_model.summary())
    print()

    # Odds ratios for easier interpretation
    or_table = pd.DataFrame(
        {
            "coef": logit_model.params,
            "odds_ratio": np.exp(logit_model.params),
            "p_value": logit_model.pvalues,
        }
    )
    print("Odds ratios:")
    print(or_table.to_string())


if __name__ == "__main__":
    main()

