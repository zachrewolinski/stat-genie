import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Group summaries
    group_means = df.groupby("children")["affairs"].mean()
    group_rates = df.groupby("children")["affair_any"].mean()

    print("Mean affairs by children status:")
    print(group_means)
    print("\nShare with any affair by children status:")
    print(group_rates)

    # Simple difference in means (OLS)
    ols_simple = smf.ols("affairs ~ C(children)", data=df).fit(cov_type="HC3")
    print("\nOLS difference in means (affairs ~ children):")
    print(ols_simple.summary().tables[1])

    # OLS with controls
    ols_controls = smf.ols(
        "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + "
        "education + occupation + rating",
        data=df,
    ).fit(cov_type="HC3")
    print("\nOLS with controls:")
    print(ols_controls.summary().tables[1])

    # Logit for any affair
    logit_controls = smf.logit(
        "affair_any ~ C(children) + C(gender) + age + yearsmarried + religiousness + "
        "education + occupation + rating",
        data=df,
    ).fit(disp=0)
    print("\nLogit (any affair) with controls:")
    print(logit_controls.summary().tables[1])

    # Poisson for count outcome
    poisson_controls = smf.poisson(
        "affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + "
        "education + occupation + rating",
        data=df,
    ).fit(disp=0)
    print("\nPoisson with controls:")
    print(poisson_controls.summary().tables[1])


if __name__ == "__main__":
    main()
