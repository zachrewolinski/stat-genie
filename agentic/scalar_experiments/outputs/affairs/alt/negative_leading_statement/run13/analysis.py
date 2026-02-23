import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
    )

    print("Descriptive statistics by children status:")
    print(desc)
    print()

    # Logistic regression for any_affair ~ children + covariates
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression results:")
    print(logit_model.summary())

    # Extract effect of having children (yes vs no)
    # With C(children), baseline is typically the first in alphabetical order ("no"),
    # so the parameter for C(children)[T.yes] reflects the contrast yes - no.
    params = logit_model.params
    pvalues = logit_model.pvalues

    child_param_name = None
    for name in params.index:
        if "C(children)" in name:
            child_param_name = name
            break

    if child_param_name is None:
        raise RuntimeError("Could not find children coefficient in the model.")

    child_coef = params[child_param_name]
    child_p = pvalues[child_param_name]

    print()
    print(f"Coefficient for {child_param_name}: {child_coef:.4f}, p-value = {child_p:.4g}")

    # Also compute average marginal effect for children for additional context
    margeff = logit_model.get_margeff(at="overall", method="dydx")
    me_df = margeff.summary_frame()
    if child_param_name in me_df.index:
        child_ame = me_df.loc[child_param_name, "dy/dx"]
        print(f"Average marginal effect for children: {child_ame:.4f}")
    else:
        print("Average marginal effect for children not found.")


if __name__ == "__main__":
    main()

