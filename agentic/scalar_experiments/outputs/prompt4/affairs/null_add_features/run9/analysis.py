import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic cleaning: ensure expected columns exist
    expected_cols = [
        "affairs",
        "gender",
        "age",
        "yearsmarried",
        "children",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Binary indicator: had any affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Drop any rows with missing values in variables we will use
    model_vars = expected_cols + ["has_affair"]
    df_model = df[model_vars].dropna().copy()

    # Group summaries by children status
    group_affairs = (
        df_model.groupby("children")["affairs"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_affairs"})
    )
    group_has_affair = df_model.groupby("children")["has_affair"].mean()

    # Two-sample t-test for mean number of affairs
    affairs_yes = df_model.loc[df_model["children"] == "yes", "affairs"]
    affairs_no = df_model.loc[df_model["children"] == "no", "affairs"]
    t_stat, p_t = stats.ttest_ind(affairs_yes, affairs_no, equal_var=False)

    # Chi-squared test for difference in probability of any affair
    ct = pd.crosstab(df_model["children"], df_model["has_affair"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

    # Logistic regression: probability of any affair
    logit_formula = (
        "has_affair ~ C(children) + gender + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df_model).fit(disp=False)
    logit_children_coef = logit_model.params.get("C(children)[T.yes]", np.nan)
    logit_children_p = logit_model.pvalues.get("C(children)[T.yes]", np.nan)

    # Poisson regression: count of affairs
    pois_formula = (
        "affairs ~ C(children) + gender + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    pois_model = smf.poisson(pois_formula, data=df_model).fit(disp=False)
    pois_children_coef = pois_model.params.get("C(children)[T.yes]", np.nan)
    pois_children_p = pois_model.pvalues.get("C(children)[T.yes]", np.nan)

    # Print key results for inspection
    print("Group means for affairs by children:")
    print(group_affairs)
    print("\nProportion with any affair by children:")
    print(group_has_affair)
    print("\nT-test (affairs_yes vs affairs_no):")
    print(f"t = {t_stat:.3f}, p = {p_t:.4g}")
    print("\nChi-squared test for has_affair vs children:")
    print(f"chi2 = {chi2:.3f}, p = {p_chi2:.4g}, dof = {dof}")
    print("\nLogistic regression children coefficient (C(children)[T.yes]):")
    print(f"coef = {logit_children_coef:.4f}, p = {logit_children_p:.4g}")
    print("\nPoisson regression children coefficient (C(children)[T.yes]):")
    print(f"coef = {pois_children_coef:.4f}, p = {pois_children_p:.4g}")


if __name__ == "__main__":
    main()

