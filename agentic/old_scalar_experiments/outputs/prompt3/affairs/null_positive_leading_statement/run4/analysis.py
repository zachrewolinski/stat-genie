import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create binary outcome: any affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives by children
    descriptives = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "prop_any_affair", "sum": "num_with_affair"})
    )
    print("Descriptives by children:\n", descriptives, "\n", flush=True)

    # Unadjusted logistic regression: any_affair ~ children
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("Unadjusted logistic regression:\n", model_unadj.summary(), "\n", flush=True)

    # Adjusted logistic regression with standard covariates
    formula_adj = (
        "any_affair ~ C(children) + gender + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model_adj = smf.logit(formula_adj, data=df).fit(disp=False)
    print("Adjusted logistic regression:\n", model_adj.summary(), "\n", flush=True)

    # Extract key results for children effect from adjusted model
    params = model_adj.params
    conf_int = model_adj.conf_int()
    odds_ratios = params.apply(lambda x: float(pd.np.exp(x)))  # type: ignore[attr-defined]

    # children is a two-level factor; statsmodels creates C(children)[T.yes] or [T.no]
    # Find coefficient name containing 'children' that is not the reference
    child_term = next(name for name in params.index if "C(children)[T." in name)

    coef = float(params[child_term])
    ci_low, ci_high = map(float, conf_int.loc[child_term])
    or_est = float(odds_ratios[child_term])

    print("\nKey effect (children term):")
    print(" term:", child_term)
    print(" log-odds coef:", coef)
    print(" 95% CI:", (ci_low, ci_high))
    print(" odds ratio:", or_est)


if __name__ == "__main__":
    main()
