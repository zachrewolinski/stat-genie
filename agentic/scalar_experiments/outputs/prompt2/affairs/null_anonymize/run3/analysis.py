import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df.columns = [
        "id",
        "affairs",
        "gender",
        "age",
        "yrs_married",
        "children",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    df["affair_any"] = (df["affairs"] > 0).astype(int)

    summary_children = (
        df.groupby("children")
        .agg(
            n=("affair_any", "size"),
            mean_affairs=("affairs", "mean"),
            prop_any=("affair_any", "mean"),
        )
        .reset_index()
    )

    print("Affair outcomes by children status:")
    print(summary_children)
    print()

    model = smf.logit(
        "affair_any ~ C(children) + age + yrs_married + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    print("Logistic regression on any affair ~ children + covariates")
    print(model.summary())
    print()
    print("Coefficients:")
    print(model.params)
    print()
    print("P-values:")
    print(model.pvalues)


if __name__ == "__main__":
    main()

