import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # In this dataset, the column named "age" is actually the affair frequency code,
    # and "religiousness" is a yes/no indicator for whether there are children.
    df = df.copy()
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["age"] > 0).astype(int)

    print("Sample size:", len(df))
    print("Children value counts:\n", df["has_children"].value_counts(dropna=False))

    # Basic group summaries
    group_means = df.groupby("has_children")["age"].mean()
    group_props = df.groupby("has_children")["any_affair"].mean()
    print("\nMean affair frequency by children (0=no,1=yes):\n", group_means)
    print("\nProportion with any affair by children (0=no,1=yes):\n", group_props)

    # Logistic regression: any affair ~ children only
    logit_simple = smf.glm(
        formula="any_affair ~ has_children",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print("\nLogistic regression (any_affair ~ has_children) summary:")
    print(logit_simple.summary())

    # Logistic regression with basic controls
    logit_controls = smf.glm(
        formula="any_affair ~ has_children + yearsmarried + rating + education + C(gender)",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print("\nLogistic regression with controls summary:")
    print(logit_controls.summary())


if __name__ == "__main__":
    main()

