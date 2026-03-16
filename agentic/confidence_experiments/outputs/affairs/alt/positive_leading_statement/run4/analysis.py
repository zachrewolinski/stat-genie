import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive stats by children
    grouped = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )

    print("Descriptives by children:")
    print(grouped.to_string(index=False))
    print()

    # Logistic regression (bivariate): probability of any affair ~ children only
    bivar_formula = "any_affair ~ C(children)"
    bivar_logit = smf.logit(formula=bivar_formula, data=df).fit(disp=False)
    print("Bivariate logistic regression (any_affair ~ C(children)):")
    print(bivar_logit.summary2())
    print()

    # Logistic regression (multivariable): probability of any affair ~ children + controls
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression results (any_affair as outcome):")
    print(logit_model.summary2())


if __name__ == "__main__":
    main()
