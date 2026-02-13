import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("=== Basic descriptives by children ===")
    grouped = df.groupby("children")["has_affair"].agg(["mean", "count"])
    print(grouped)
    print()

    print("=== Mean affairs score by children ===")
    print(df.groupby("children")["affairs"].mean())
    print()

    # Logistic regression controlling for observed covariates
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("=== Logistic regression summary ===")
    print(model.summary())
    print()

    # Show exponentiated coefficients (odds ratios)
    params = model.params
    conf = model.conf_int()
    or_table = params.to_frame("coef")
    or_table["odds_ratio"] = params.apply(lambda x: float(pd.np.exp(x)))
    or_table["ci_lower"] = conf[0].apply(lambda x: float(pd.np.exp(x)))
    or_table["ci_upper"] = conf[1].apply(lambda x: float(pd.np.exp(x)))

    print("=== Odds ratios ===")
    print(or_table)


if __name__ == "__main__":
    main()

