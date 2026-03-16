import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create binary outcome: any affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive stats by children status
    grouped = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("any_affair", "mean"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(grouped.to_string(index=False))
    print()

    # Logistic regression: any_affair ~ children + controls
    # Children is categorical (yes/no); use C(children) so that "no" is baseline.
    formula_simple = "any_affair ~ C(children)"
    formula_controls = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    print("Logistic regression without controls:")
    model_simple = smf.logit(formula_simple, data=df).fit(disp=False)
    print(model_simple.summary())
    print()

    # Odds ratio for children[T.yes]
    params_simple = model_simple.params
    conf_simple = model_simple.conf_int()
    if "C(children)[T.yes]" in params_simple.index:
        beta_child = params_simple["C(children)[T.yes]"]
        or_child = float(np.exp(beta_child))
        ci_low, ci_high = np.exp(conf_simple.loc["C(children)[T.yes]"])
        pvalue_child = model_simple.pvalues["C(children)[T.yes]"]
        print("Effect of having children (simple model):")
        print(f"  log-odds = {beta_child:.3f}")
        print(f"  odds ratio = {or_child:.3f}")
        print(f"  95% CI for OR = [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  p-value = {pvalue_child:.4g}")
        print()

    print("Logistic regression with controls:")
    model_controls = smf.logit(formula_controls, data=df).fit(disp=False)
    print(model_controls.summary())
    print()

    params_ctrl = model_controls.params
    conf_ctrl = model_controls.conf_int()
    if "C(children)[T.yes]" in params_ctrl.index:
        beta_child_ctrl = params_ctrl["C(children)[T.yes]"]
        or_child_ctrl = float(np.exp(beta_child_ctrl))
        ci_low_ctrl, ci_high_ctrl = np.exp(conf_ctrl.loc["C(children)[T.yes]"])
        pvalue_child_ctrl = model_controls.pvalues["C(children)[T.yes]"]
        print("Effect of having children (with controls):")
        print(f"  log-odds = {beta_child_ctrl:.3f}")
        print(f"  odds ratio = {or_child_ctrl:.3f}")
        print(f"  95% CI for OR = [{ci_low_ctrl:.3f}, {ci_high_ctrl:.3f}]")
        print(f"  p-value = {pvalue_child_ctrl:.4g}")


if __name__ == "__main__":
    main()

