import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # According to info.json, column names are shuffled.
    # Map them to their semantic meanings for this analysis:
    # - age: frequency of extramarital intercourse in past year (0, 1, 2, 3, 7, 12, ...)
    # - religiousness: "yes"/"no" indicating whether there are children in the marriage
    df["extramarital_freq"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Basic group-wise descriptives
    grouped = df.groupby("has_children")["extramarital_freq"]
    print("Mean extramarital frequency by has_children (1=yes, 0=no):")
    print(grouped.mean())
    print()

    print("Std dev extramarital frequency by has_children (1=yes, 0=no):")
    print(grouped.std())
    print()

    # Binary outcome: any affair vs none
    df["any_affair"] = (df["extramarital_freq"] > 0).astype(int)
    prop_any = df.groupby("has_children")["any_affair"].mean()
    print("Proportion with any affair by has_children (1=yes, 0=no):")
    print(prop_any)
    print()

    # Nonparametric comparison of frequency distributions
    no_children = df.loc[df["has_children"] == 0, "extramarital_freq"]
    yes_children = df.loc[df["has_children"] == 1, "extramarital_freq"]
    u_stat, p_val = stats.mannwhitneyu(
        no_children, yes_children, alternative="two-sided"
    )
    print(f"Mann-Whitney U test (freq, children vs no-children): U={u_stat:.3f}, p={p_val:.5g}")
    print()

    # Simple logistic regression: any_affair ~ has_children
    X = df[["has_children"]].copy()
    X = sm.add_constant(X, has_constant="add")
    logit_model = sm.Logit(df["any_affair"], X).fit(disp=False)
    coef = logit_model.params["has_children"]
    p = logit_model.pvalues["has_children"]
    or_val = float(np.exp(coef))
    print(
        "Logit(any_affair) ~ has_children:"
        f" coef={coef:.3f}, OR={or_val:.3f}, p={p:.5g}"
    )
    print()

    # Build additional covariates based on semantic descriptions in info.json
    df["age_years"] = df["occupation"]  # age band
    df["years_married"] = df["children"]
    df["religiousness_score"] = df["rating"]
    df["education_years"] = df["yearsmarried"]
    df["occupation_code"] = df["rownames"]
    df["marriage_rating"] = df["affairs"]

    covariates = [
        "has_children",
        "age_years",
        "years_married",
        "religiousness_score",
        "education_years",
        "occupation_code",
        "marriage_rating",
    ]

    X2 = df[covariates].copy()
    X2 = sm.add_constant(X2, has_constant="add")
    logit_model_adj = sm.Logit(df["any_affair"], X2).fit(disp=False)
    coef2 = logit_model_adj.params["has_children"]
    p2 = logit_model_adj.pvalues["has_children"]
    or2 = float(np.exp(coef2))
    print(
        "Adjusted logit(any_affair) ~ has_children + controls:"
        f" coef={coef2:.3f}, OR={or2:.3f}, p={p2:.5g}"
    )


if __name__ == "__main__":
    main()

