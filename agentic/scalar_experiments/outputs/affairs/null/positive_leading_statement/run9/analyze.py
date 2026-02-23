import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic summaries by children status
    summary = (
        df.groupby("children")["affairs"]
        .agg(["mean", "std", "count"])
        .rename_axis("children")
        .reset_index()
    )
    print("Affair frequency by children status:")
    print(summary.to_string(index=False))
    print()

    # Binary outcome: any affair vs. none
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Unadjusted comparison of proportions
    prop = (
        df.groupby("children")["any_affair"]
        .mean()
        .rename("prop_any_affair")
        .reset_index()
    )
    print("Proportion with any affair by children status:")
    print(prop.to_string(index=False))
    print()

    # Logistic regression for any affair ~ children + covariates
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression for any affair (1) vs none (0):")
    print(logit_model.summary())
    print()

    # Extract key coefficient for children
    params = logit_model.params
    conf_int = logit_model.conf_int()
    odds_ratios = params.apply(lambda x: float(pd.np.exp(x)))

    # children is binary yes/no, smf with C(children) uses baseline alphabetically, likely 'no'.
    # We locate the coefficient that compares yes to no.
    child_coef_name = [name for name in params.index if "C(children)" in name]
    if child_coef_name:
        name = child_coef_name[0]
        coef = params[name]
        ci_low, ci_high = conf_int.loc[name]
        or_val = odds_ratios[name]
        or_low = float(pd.np.exp(ci_low))
        or_high = float(pd.np.exp(ci_high))
        print(f"Children coefficient ({name}): {coef:.3f}")
        print(f"Odds ratio: {or_val:.3f} (95% CI: {or_low:.3f}, {or_high:.3f})")
    else:
        print("Children coefficient not found in model parameters.")


if __name__ == "__main__":
    main()

