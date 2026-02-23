import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome measures
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic descriptives
    group = df.groupby("feature6")
    mean_freq = group["feature2"].mean()
    prob_any = group["any_affair"].mean()

    print("Mean affair frequency by children (feature6):")
    for k, v in mean_freq.items():
        print(f"  {k}: {v:.3f}")

    print("\nProportion with any affair by children (feature6):")
    for k, v in prob_any.items():
        print(f"  {k}: {v:.3f}")

    # Logistic regression: any affair ~ children (unadjusted)
    model_unadj = smf.logit("any_affair ~ C(feature6)", data=df).fit(disp=False)
    coef_children_unadj = model_unadj.params.get("C(feature6)[T.yes]")
    pval_children_unadj = model_unadj.pvalues.get("C(feature6)[T.yes]")
    odds_ratio_unadj = np.exp(coef_children_unadj)

    print("\nLogistic regression (unadjusted): any_affair ~ C(feature6)")
    print(f"  Coefficient for children=yes vs no: {coef_children_unadj:.4f}")
    print(f"  Odds ratio: {odds_ratio_unadj:.4f}")
    print(f"  p-value: {pval_children_unadj:.4g}")

    # Logistic regression: any affair ~ children + covariates
    formula_adj = (
        "any_affair ~ C(feature6) + C(feature3) + "
        "feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    coef_children_adj = model_adj.params.get("C(feature6)[T.yes]")
    pval_children_adj = model_adj.pvalues.get("C(feature6)[T.yes]")
    odds_ratio_adj = np.exp(coef_children_adj)

    print(
        "\nLogistic regression (adjusted): any_affair ~ C(feature6) + "
        "C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )
    print(f"  Coefficient for children=yes vs no: {coef_children_adj:.4f}")
    print(f"  Odds ratio: {odds_ratio_adj:.4f}")
    print(f"  p-value: {pval_children_adj:.4g}")


if __name__ == "__main__":
    main()
