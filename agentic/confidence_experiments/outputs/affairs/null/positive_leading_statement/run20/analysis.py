import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children indicator: 1 if there are children in the marriage
    df["children_yes"] = (df["children"] == "yes").astype(int)

    print("=== Basic prevalence of affairs by children status ===")
    prev = df.groupby("children")["has_affair"].agg(["mean", "sum", "count"])
    print(prev)
    print()

    # Unadjusted logistic regression: any affair ~ children_yes
    print("=== Unadjusted logistic regression: has_affair ~ children_yes ===")
    model_unadj = smf.logit("has_affair ~ children_yes", data=df).fit(
        disp=False, maxiter=200
    )
    print(model_unadj.summary())
    params_u = model_unadj.params
    conf_u = model_unadj.conf_int()
    or_u = float(np.exp(params_u["children_yes"]))
    ci_low_u = float(np.exp(conf_u.loc["children_yes", 0]))
    ci_high_u = float(np.exp(conf_u.loc["children_yes", 1]))
    p_u = float(model_unadj.pvalues["children_yes"])
    print(
        f"Unadjusted OR for children_yes: {or_u:.3f} "
        f"(95% CI {ci_low_u:.3f}-{ci_high_u:.3f}), p={p_u:.4g}"
    )
    print()

    # Adjusted logistic regression including key covariates
    print(
        "=== Adjusted logistic regression: has_affair ~ children_yes + "
        "C(gender) + age + yearsmarried + religiousness + education + "
        "occupation + rating ==="
    )
    formula_adj = (
        "has_affair ~ children_yes + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False, maxiter=200)
    print(model_adj.summary())
    params_a = model_adj.params
    conf_a = model_adj.conf_int()
    or_a = float(np.exp(params_a["children_yes"]))
    ci_low_a = float(np.exp(conf_a.loc["children_yes", 0]))
    ci_high_a = float(np.exp(conf_a.loc["children_yes", 1]))
    p_a = float(model_adj.pvalues["children_yes"])
    print(
        f"Adjusted OR for children_yes: {or_a:.3f} "
        f"(95% CI {ci_low_a:.3f}-{ci_high_a:.3f}), p={p_a:.4g}"
    )
    print()

    print("=== Distribution of affair counts by children status ===")
    counts = df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"])
    print(counts)


if __name__ == "__main__":
    main()

