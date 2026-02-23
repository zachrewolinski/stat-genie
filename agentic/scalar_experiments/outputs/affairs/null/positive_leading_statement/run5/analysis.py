import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_ind"] = (df["children"] == "yes").astype(int)
    df["male"] = (df["gender"] == "male").astype(int)

    print("Dataset shape:", df.shape)
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    print("\nAffairs count summary by children:")
    print(df.groupby("children")["affairs"].describe())

    print("\nAny-affair (affairs > 0) rate by children:")
    any_rates = df.groupby("children")["any_affair"].mean()
    print(any_rates)

    # Two-proportion z-test for any affair between those with and without children
    counts = df.groupby("children")["any_affair"].sum().loc[["no", "yes"]]
    nobs = df.groupby("children")["any_affair"].count().loc[["no", "yes"]]
    stat, pval = proportions_ztest(counts, nobs)
    print(
        "\nTwo-proportion z-test for any affair (no vs yes children):"
        f"\n  z-statistic = {stat:.3f}, p-value = {pval:.4f}"
    )
    print("  counts (no, yes)   =", tuple(counts.values))
    print("  nobs   (no, yes)   =", tuple(nobs.values))
    print("  rates  (no, yes)   =", tuple((counts / nobs).values))

    # Logistic regression for any affair with children and controls
    formula_logit = (
        "any_affair ~ children_ind + age + yearsmarried + religiousness "
        "+ education + occupation + rating + male"
    )
    logit_model = smf.logit(formula_logit, data=df).fit(disp=False)
    print("\nLogistic regression results (any_affair ~ children + controls):")
    print(logit_model.summary())

    # Odds ratio and 95% CI for children indicator
    params = logit_model.params
    conf = logit_model.conf_int()
    child_coef = params["children_ind"]
    child_or = float(np.exp(child_coef))
    child_ci_low, child_ci_high = np.exp(conf.loc["children_ind"])
    print(
        "\nChildren effect (logistic model):"
        f"\n  log-odds coef = {child_coef:.3f}"
        f"\n  odds ratio    = {child_or:.3f}"
        f"\n  95% CI OR     = ({child_ci_low:.3f}, {child_ci_high:.3f})"
        f"\n  p-value       = {logit_model.pvalues['children_ind']:.4f}"
    )

    # Poisson regression for affair counts with the same controls
    formula_pois = (
        "affairs ~ children_ind + age + yearsmarried + religiousness "
        "+ education + occupation + rating + male"
    )
    pois_model = smf.poisson(formula_pois, data=df).fit(disp=False)
    print("\nPoisson regression results (affairs ~ children + controls):")
    print(pois_model.summary())

    pois_params = pois_model.params
    pois_conf = pois_model.conf_int()
    child_coef_pois = pois_params["children_ind"]
    child_rr = float(np.exp(child_coef_pois))
    child_rr_low, child_rr_high = np.exp(pois_conf.loc["children_ind"])
    print(
        "\nChildren effect (Poisson model):"
        f"\n  log-rate coef  = {child_coef_pois:.3f}"
        f"\n  rate ratio     = {child_rr:.3f}"
        f"\n  95% CI RR      = ({child_rr_low:.3f}, {child_rr_high:.3f})"
        f"\n  p-value        = {pois_model.pvalues['children_ind']:.4f}"
    )


if __name__ == "__main__":
    main()

