import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In this dataset, the column named "religiousness" actually encodes
    # whether there are children in the marriage (values: "yes"/"no").
    df["has_children"] = (df["religiousness"] == "yes").astype(int)

    # The column named "age" encodes the frequency of extramarital sexual
    # intercourse in the past year (0, 1, 2, 3, 7, 12).
    df["any_affair"] = (df["age"] > 0).astype(int)

    # Basic summaries
    n_total = len(df)
    n_children = df["has_children"].sum()
    n_no_children = n_total - n_children

    prop_affair_children = df.loc[df["has_children"] == 1, "any_affair"].mean()
    prop_affair_no_children = df.loc[df["has_children"] == 0, "any_affair"].mean()

    mean_freq_children = df.loc[df["has_children"] == 1, "age"].mean()
    mean_freq_no_children = df.loc[df["has_children"] == 0, "age"].mean()

    # Two-sample z-test for difference in proportions of having any affair.
    count = np.array(
        [
            df.loc[df["has_children"] == 0, "any_affair"].sum(),
            df.loc[df["has_children"] == 1, "any_affair"].sum(),
        ]
    )
    nobs = np.array([n_no_children, n_children])
    z_stat, p_value = proportions_ztest(count, nobs)

    # Logistic regression: any_affair ~ has_children
    logit_model = smf.logit("any_affair ~ has_children", data=df).fit(disp=0)
    coef_children = logit_model.params["has_children"]
    p_children = logit_model.pvalues["has_children"]
    odds_ratio_children = float(np.exp(coef_children))

    # Print key results needed to form the conclusion.
    print("N total:", n_total)
    print("N with children:", n_children)
    print("N without children:", n_no_children)
    print("Proportion any affair (no children):", prop_affair_no_children)
    print("Proportion any affair (children):", prop_affair_children)
    print("Mean affair frequency (no children):", mean_freq_no_children)
    print("Mean affair frequency (children):", mean_freq_children)
    print("Z stat (proportion test):", z_stat)
    print("P-value (proportion test, two-sided):", p_value)
    print("Logit coef(has_children):", coef_children)
    print("Logit p-value(has_children):", p_children)
    print("Logit odds ratio(has_children):", odds_ratio_children)


if __name__ == "__main__":
    main()

