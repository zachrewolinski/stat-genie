import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In the provided metadata, the column named "age" actually codes
    # frequency of extramarital intercourse in the past year, and
    # "religiousness" is a yes/no indicator for whether there are children.
    affairs_freq = df["age"]
    has_children = df["religiousness"].map({"yes": 1, "no": 0})

    # Binary indicator: any extramarital affair in the past year.
    any_affair = (affairs_freq > 0).astype(int)

    # Simple descriptive stats by children status.
    summary = (
        pd.DataFrame(
            {
                "n": any_affair.groupby(has_children).size(),
                "prop_any_affair": any_affair.groupby(has_children).mean(),
                "mean_affair_freq": affairs_freq.groupby(has_children).mean(),
            }
        )
        .rename(index={0: "no_children", 1: "has_children"})
        .reset_index(names="children_group")
    )

    # Logistic regression for any affair ~ children + covariates.
    df_model = df.copy()
    df_model["any_affair"] = any_affair
    df_model["has_children"] = has_children

    # Exclude the arbitrary row index column if present.
    covariates = [
        "has_children",
        "education",
        "occupation",
        "children",
        "rating",
        "yearsmarried",
        "gender",
    ]
    formula = "any_affair ~ " + " + ".join(
        [c for c in covariates if c in df_model.columns]
    )

    # Treat gender as categorical if it exists.
    if "gender" in df_model.columns:
        df_model["gender"] = df_model["gender"].astype("category")

    logit_model = smf.logit(formula=formula, data=df_model).fit(disp=False)
    logit_params = logit_model.params
    logit_pvalues = logit_model.pvalues

    has_children_coef = float(logit_params["has_children"])
    has_children_pvalue = float(logit_pvalues["has_children"])
    has_children_or = float(np.exp(has_children_coef))

    # Chi-square test for independence between children and any affair.
    contingency = pd.crosstab(has_children, any_affair)
    chi2, chi_pvalue, _, _ = stats.chi2_contingency(contingency)

    # Two-sample t-test for mean affair frequency by children status.
    freq_no_children = affairs_freq[has_children == 0]
    freq_has_children = affairs_freq[has_children == 1]
    t_stat, t_pvalue = stats.ttest_ind(
        freq_no_children, freq_has_children, equal_var=False
    )

    print("Summary by children group:")
    print(summary.to_string(index=False))
    print("\nLogistic regression (any_affair ~ has_children + covariates):")
    print(f"Coefficient for has_children: {has_children_coef:.4f}")
    print(f"Odds ratio for has_children: {has_children_or:.4f}")
    print(f"P-value for has_children: {has_children_pvalue:.4g}")
    print("\nChi-square test for any_affair vs has_children:")
    print(f"Chi2 statistic: {chi2:.4f}, p-value: {chi_pvalue:.4g}")
    print("\nT-test for affair frequency by children status:")
    print(f"t statistic: {t_stat:.4f}, p-value: {t_pvalue:.4g}")


if __name__ == "__main__":
    main()
