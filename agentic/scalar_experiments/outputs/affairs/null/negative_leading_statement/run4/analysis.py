import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic cleaning / derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children
    grouped = (
        df.groupby("children")
        .agg(
            n=("affairs", "size"),
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("has_affair", "mean"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status:")
    print(grouped.to_string(index=False))
    print()

    # Logistic regression: any affair ~ children only
    logit_simple = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression (has_affair ~ C(children)):")
    print(logit_simple.summary())
    print()

    # Logistic regression with key covariates to adjust for confounding
    formula_full = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    print("Logistic regression with covariates:")
    print(logit_full.summary())
    print()

    # Marginal effect of children from the full model
    marg_eff = logit_full.get_margeff(at="overall", method="dydx")
    print("Average marginal effects from full model:")
    print(marg_eff.summary())


if __name__ == "__main__":
    main()

