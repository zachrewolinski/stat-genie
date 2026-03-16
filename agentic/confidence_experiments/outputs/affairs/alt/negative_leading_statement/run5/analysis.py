import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome variants
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries
    group_counts = df.groupby("children")["affairs"].agg(["mean", "std", "count"])
    group_props = df.groupby("children")["has_affair"].mean()

    # Two-sample t-test on affair counts (Welch)
    yes_vals = df.loc[df["children"] == "yes", "affairs"]
    no_vals = df.loc[df["children"] == "no", "affairs"]
    t_res = stats.ttest_ind(yes_vals, no_vals, equal_var=False)

    # Logistic regression for any affair
    logit_formula = (
        "has_affair ~ children + gender + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)

    # Poisson regression for affair counts
    poisson_formula = (
        "affairs ~ children + gender + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    poisson_model = smf.poisson(formula=poisson_formula, data=df).fit(disp=False)

    print("=== Group summaries by children ===")
    print(group_counts)
    print("\nProportion with any affair by children:")
    print(group_props)

    print("\n=== Welch t-test on affair counts (children yes vs no) ===")
    print(f"t = {t_res.statistic:.3f}, p = {t_res.pvalue:.4g}")

    print("\n=== Logistic regression: has_affair ~ children + covariates ===")
    print(logit_model.summary2())

    print("\n=== Poisson regression: affairs ~ children + covariates ===")
    print(poisson_model.summary2())

    # Extract key coefficients for children effect
    def extract_effect(model, param_name: str):
        params = model.params
        conf_int = model.conf_int()
        pvalues = model.pvalues
        if param_name not in params.index:
            return None
        coef = params[param_name]
        ci_low, ci_high = conf_int.loc[param_name]
        pval = pvalues[param_name]
        return coef, ci_low, ci_high, pval

    logit_children = extract_effect(logit_model, "children[T.yes]")
    poisson_children = extract_effect(poisson_model, "children[T.yes]")

    print("\n=== Extracted children effects ===")
    print("Logistic (has_affair):", logit_children)
    print("Poisson (affairs count):", poisson_children)


if __name__ == "__main__":
    main()

