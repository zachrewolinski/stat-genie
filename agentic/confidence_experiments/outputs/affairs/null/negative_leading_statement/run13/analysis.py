import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having had any extramarital affair in the last year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("=== Descriptive statistics by children ===")
    group_counts = df.groupby("children")["affairs"].agg(["mean", "median", "std", "count"])
    print(group_counts)
    print()

    print("=== Proportion with any affair by children ===")
    prop_affair = df.groupby("children")["has_affair"].mean()
    print(prop_affair)
    print()

    print("=== Logistic regression: has_affair ~ children + covariates ===")
    logit_model = smf.logit(
        "has_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + rating + C(gender) + occupation",
        data=df,
    ).fit(disp=0)
    print(logit_model.summary())
    print()

    print("=== Poisson regression: affairs count ~ children + covariates ===")
    poisson_model = smf.glm(
        "affairs ~ C(children) + age + yearsmarried + religiousness + "
        "education + rating + C(gender) + occupation",
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    print(poisson_model.summary())


if __name__ == "__main__":
    main()

