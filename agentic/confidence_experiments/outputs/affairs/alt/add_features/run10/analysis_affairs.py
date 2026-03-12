import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Define binary outcome: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic counts by children status
    counts = df.groupby("children")["has_affair"].agg(
        n="count", n_affair="sum"
    )
    counts["prop_affair"] = counts["n_affair"] / counts["n"]

    print("Counts and proportions of any affair by children status:")
    print(counts)
    print()

    # Two-proportion z-test: children=yes vs no
    props = counts["n_affair"].to_numpy()
    ns = counts["n"].to_numpy()

    if len(props) == 2:
        stat, pval = proportions_ztest(props, ns)
        print("Two-proportion z-test (any affair ~ children yes vs no):")
        print(f"  z-statistic = {stat:.3f}, p-value = {pval:.4g}")
        print()

    # Unadjusted logistic regression
    logit_unadj = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print("Unadjusted logistic regression: has_affair ~ C(children)")
    print(logit_unadj.summary())
    print()

    # Extract odds ratio for children=yes (relative to no)
    params_unadj = logit_unadj.params
    conf_unadj = logit_unadj.conf_int()
    if "C(children)[T.yes]" in params_unadj.index:
        beta = params_unadj["C(children)[T.yes]"]
        or_est = np.exp(beta)
        ci_low, ci_high = np.exp(conf_unadj.loc["C(children)[T.yes]"])
        pval_children = logit_unadj.pvalues["C(children)[T.yes]"]
        print("Unadjusted effect of children=yes:")
        print(
            f"  OR = {or_est:.3f} "
            f"(95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
            f"p-value = {pval_children:.4g}"
        )
        print()

    # Adjusted logistic regression controlling for key covariates
    # (using original Fair affairs predictors)
    formula_adj = (
        "has_affair ~ C(children) + age + yearsmarried "
        "+ religiousness + education + rating + C(gender)"
    )
    logit_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print("Adjusted logistic regression:")
    print(formula_adj)
    print(logit_adj.summary())
    print()

    params_adj = logit_adj.params
    conf_adj = logit_adj.conf_int()
    if "C(children)[T.yes]" in params_adj.index:
        beta_adj = params_adj["C(children)[T.yes]"]
        or_adj = np.exp(beta_adj)
        ci_low_adj, ci_high_adj = np.exp(conf_adj.loc["C(children)[T.yes]"])
        pval_children_adj = logit_adj.pvalues["C(children)[T.yes]"]
        print("Adjusted effect of children=yes:")
        print(
            f"  OR = {or_adj:.3f} "
            f"(95% CI [{ci_low_adj:.3f}, {ci_high_adj:.3f}]), "
            f"p-value = {pval_children_adj:.4g}"
        )


if __name__ == "__main__":
    main()

