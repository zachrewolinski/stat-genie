import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("affairs.csv")

    # Basic cleaning
    df = df.copy()
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive stats by children
    grp = df.groupby("children")
    desc = grp["affairs"].agg(["count", "mean", "median", "std"]).rename(columns={"count": "n"})
    prop_any = grp["any_affair"].mean().rename("prop_any_affair")
    summary = desc.join(prop_any)

    print("Descriptive stats by children:")
    print(summary)
    print()

    # Two-sample t-test for mean affairs
    affairs_yes = df.loc[df["children_yes"] == 1, "affairs"].values
    affairs_no = df.loc[df["children_yes"] == 0, "affairs"].values
    t_stat, p_val, dfree = ttest_ind(affairs_yes, affairs_no, usevar="unequal")
    print("T-test (unequal var) for mean affairs (children yes vs no):")
    print({"t_stat": float(t_stat), "p_value": float(p_val), "df": float(dfree)})
    print()

    # Proportion test for any affair
    counts = np.array([
        df.loc[df["children_yes"] == 1, "any_affair"].sum(),
        df.loc[df["children_yes"] == 0, "any_affair"].sum(),
    ])
    nobs = np.array([
        (df["children_yes"] == 1).sum(),
        (df["children_yes"] == 0).sum(),
    ])
    z_stat, p_prop = proportions_ztest(counts, nobs)
    print("Proportion z-test for any affair (children yes vs no):")
    print({"z_stat": float(z_stat), "p_value": float(p_prop)})
    print()

    # Poisson regression (with controls) for affairs count
    # Use gender as categorical; drop rows with missing values in used columns
    model_df = df[[
        "affairs", "children_yes", "age", "yearsmarried", "rating",
        "religiousness", "education", "occupation", "gender"
    ]].dropna()

    poisson = smf.glm(
        "affairs ~ children_yes + age + yearsmarried + rating + religiousness + education + occupation + C(gender)",
        data=model_df,
        family=sm.families.Poisson(),
    ).fit()

    coef = poisson.params["children_yes"]
    irr = float(np.exp(coef))
    p_child = float(poisson.pvalues["children_yes"])

    print("Poisson regression (count of affairs) with controls:")
    print({
        "children_yes_coef": float(coef),
        "children_yes_IRR": irr,
        "children_yes_p_value": p_child,
    })


if __name__ == "__main__":
    main()
