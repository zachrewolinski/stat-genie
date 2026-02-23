import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by presence of children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_affair=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )

    print("Descriptive stats by children:")
    print(desc.to_string(index=False))
    print()

    # Logistic regression for having any affair, with controls
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression results (has_affair as outcome):")
    print(logit_model.summary())
    print()

    # Odds ratios for easier interpretation
    or_table = pd.DataFrame(
        {
            "OR": np.exp(logit_model.params),
            "p_value": logit_model.pvalues,
        }
    )
    print("Odds ratios and p-values:")
    print(or_table.to_string())


if __name__ == "__main__":
    main()

