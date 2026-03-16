import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for engaging in any extramarital affair in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Encode having children as 1/0 for regression
    df["child_yes"] = (df["children"] == "yes").astype(int)

    # Group differences: affair count and probability of any affair
    group_summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("affair_any", "mean"),
            n=("affair_any", "size"),
        )
        .reset_index()
    )

    # 2x2 table and chi-square test for association between children and any affair
    contingency = pd.crosstab(df["children"], df["affair_any"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Unadjusted logistic regression: any affair ~ children
    model_unadj = smf.logit("affair_any ~ child_yes", data=df).fit(disp=False)
    coef_child_unadj = model_unadj.params["child_yes"]
    se_child_unadj = model_unadj.bse["child_yes"]
    p_child_unadj = model_unadj.pvalues["child_yes"]
    or_child_unadj = float(np.exp(coef_child_unadj))

    # Adjusted logistic regression controlling for key covariates
    formula_adj = (
        "affair_any ~ child_yes + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    coef_child_adj = model_adj.params["child_yes"]
    se_child_adj = model_adj.bse["child_yes"]
    p_child_adj = model_adj.pvalues["child_yes"]
    or_child_adj = float(np.exp(coef_child_adj))

    print("=== Group summaries by children ===")
    print(group_summary.to_string(index=False))
    print()

    print("=== Contingency table: children x any affair ===")
    print(contingency)
    print("Chi-square statistic:", chi2)
    print("Chi-square p-value:", p_chi2)
    print()

    print("=== Unadjusted logistic regression: affair_any ~ child_yes ===")
    print(model_unadj.summary())
    print(f"Odds ratio for child_yes: {or_child_unadj:.3f}")
    print(
        f"coef={coef_child_unadj:.4f}, se={se_child_unadj:.4f}, "
        f"p={p_child_unadj:.4f}"
    )
    print()

    print("=== Adjusted logistic regression ===")
    print(model_adj.summary())
    print(f"Odds ratio for child_yes (adjusted): {or_child_adj:.3f}")
    print(
        f"coef={coef_child_adj:.4f}, se={se_child_adj:.4f}, "
        f"p={p_child_adj:.4f}"
    )


if __name__ == "__main__":
    main()

