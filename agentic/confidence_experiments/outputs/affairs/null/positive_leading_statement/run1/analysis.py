import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_binary"] = (df["children"] == "yes").astype(int)

    print("=== Descriptive statistics by children ===")
    grouped = df.groupby("children")
    desc = grouped["affairs"].agg(["count", "mean"])
    any_affair_rate = grouped["any_affair"].mean()
    print(desc)
    print("\nProportion with any affair (affairs>0):")
    print(any_affair_rate)

    print("\n=== Logistic regression: any_affair ~ children + covariates ===")
    logit_formula = (
        "any_affair ~ children_binary + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    print(logit_model.summary())

    # Odds ratio for children_binary
    params = logit_model.params
    conf = logit_model.conf_int()
    if "children_binary" in params.index:
        or_children = np.exp(params["children_binary"])
        ci_low = np.exp(conf.loc["children_binary", 0])
        ci_high = np.exp(conf.loc["children_binary", 1])
        pvalue = logit_model.pvalues["children_binary"]
        print("\nChildren (binary) odds ratio for any affair:")
        print(f"OR={or_children:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f}), p={pvalue:.4f}")

    print("\n=== Poisson regression: affairs count ~ children + covariates ===")
    poisson_formula = (
        "affairs ~ children_binary + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    poisson_model = smf.poisson(poisson_formula, data=df).fit(disp=False)
    print(poisson_model.summary())

    if "children_binary" in poisson_model.params.index:
        irr_children = np.exp(poisson_model.params["children_binary"])
        ci_low_p, ci_high_p = np.exp(poisson_model.conf_int().loc["children_binary"])
        pvalue_p = poisson_model.pvalues["children_binary"]
        print("\nChildren (binary) incidence-rate ratio for affair count:")
        print(f"IRR={irr_children:.3f}, 95% CI=({ci_low_p:.3f}, {ci_high_p:.3f}), p={pvalue_p:.4f}")


if __name__ == "__main__":
    main()

