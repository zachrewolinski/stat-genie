import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the last year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives: affair rate by presence of children
    summary = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "affair_rate", "count": "n", "sum": "n_with_affair"})
    )

    print("Affair rate by children status:")
    print(summary)
    print()

    # Unadjusted logistic regression: any_affair ~ children
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    params_unadj = model_unadj.params
    conf_unadj = model_unadj.conf_int()
    pvalues_unadj = model_unadj.pvalues

    print("Unadjusted logistic regression: any_affair ~ C(children)")
    print(model_unadj.summary())
    print()

    # Extract effect of having children (yes vs no)
    coef_children = params_unadj.get("C(children)[T.yes]", np.nan)
    conf_children = conf_unadj.loc["C(children)[T.yes]"].to_list() if "C(children)[T.yes]" in conf_unadj.index else [np.nan, np.nan]
    p_children = pvalues_unadj.get("C(children)[T.yes]", np.nan)

    if not np.isnan(coef_children):
        odds_ratio = float(np.exp(coef_children))
        ci_lower = float(np.exp(conf_children[0]))
        ci_upper = float(np.exp(conf_children[1]))
        print("Effect of having children (yes vs no):")
        print(f"  Log-odds coefficient: {coef_children:.4f}")
        print(f"  Odds ratio: {odds_ratio:.4f}")
        print(f"  95% CI for odds ratio: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  p-value: {p_children:.4g}")
        print()

    # Adjusted logistic regression including key covariates
    formula_adj = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)

    print("Adjusted logistic regression:")
    print(model_adj.summary())
    print()

    params_adj = model_adj.params
    conf_adj = model_adj.conf_int()
    pvalues_adj = model_adj.pvalues

    coef_children_adj = params_adj.get("C(children)[T.yes]", np.nan)
    conf_children_adj = conf_adj.loc["C(children)[T.yes]"].to_list() if "C(children)[T.yes]" in conf_adj.index else [np.nan, np.nan]
    p_children_adj = pvalues_adj.get("C(children)[T.yes]", np.nan)

    if not np.isnan(coef_children_adj):
        odds_ratio_adj = float(np.exp(coef_children_adj))
        ci_lower_adj = float(np.exp(conf_children_adj[0]))
        ci_upper_adj = float(np.exp(conf_children_adj[1]))
        print("Adjusted effect of having children (yes vs no):")
        print(f"  Log-odds coefficient: {coef_children_adj:.4f}")
        print(f"  Odds ratio: {odds_ratio_adj:.4f}")
        print(f"  95% CI for odds ratio: [{ci_lower_adj:.4f}, {ci_upper_adj:.4f}]")
        print(f"  p-value: {p_children_adj:.4g}")
        print()


if __name__ == "__main__":
    main()

