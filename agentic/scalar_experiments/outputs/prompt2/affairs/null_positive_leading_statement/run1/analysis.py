import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having at least one extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    print("Descriptive statistics by children (yes/no):")
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_with_affair=("has_affair", "mean"),
            count=("has_affair", "size"),
        )
        .reset_index()
    )
    print(desc.to_string(index=False))

    # Unadjusted logistic regression: any affair ~ children
    print("\nUnadjusted logistic regression: has_affair ~ C(children)")
    model_unadj = smf.logit("has_affair ~ C(children)", data=df).fit(disp=False)
    print(model_unadj.summary())

    # Adjusted logistic regression controlling for key covariates
    print(
        "\nAdjusted logistic regression: "
        "has_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    model_adj = smf.logit(
        "has_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    print(model_adj.summary())

    # Predicted probabilities for having an affair by children status
    print("\nAdjusted predicted probability of any affair by children status:")
    for children_value in ["no", "yes"]:
        exog = df.copy()
        exog["children"] = children_value
        preds = model_adj.predict(exog)
        print(
            f" children = {children_value:3s} -> "
            f"mean predicted P(affair) = {preds.mean():.3f}"
        )


if __name__ == "__main__":
    main()

