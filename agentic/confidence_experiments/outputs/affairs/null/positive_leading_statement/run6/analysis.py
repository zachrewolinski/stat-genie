import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic transformation: create a binary indicator for any affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive stats by children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )
    print("Descriptive statistics by children:")
    print(desc.to_string(index=False))

    # Linear regression for affair frequency
    ols_model = smf.ols(
        "affairs ~ C(children) + age + yearsmarried + religiousness + education + C(occupation) + rating",
        data=df,
    ).fit()
    print("\nOLS regression on affair frequency:")
    print(ols_model.summary())

    # Logistic regression for any affair
    logit_model = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + education + C(occupation) + rating",
        data=df,
    ).fit(disp=False)
    print("\nLogistic regression on any affair:")
    print(logit_model.summary())


if __name__ == "__main__":
    main()

