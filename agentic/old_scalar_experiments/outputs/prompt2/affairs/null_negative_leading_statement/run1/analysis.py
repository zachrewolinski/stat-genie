import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any affairs in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives by children
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    print("Descriptives by children:")
    print(desc.to_string(index=False))
    print()

    # Logistic regression for any affair, controlling for other covariates
    # Use treatment coding with children == "no" as reference
    model_df = df.copy()
    model_df["children_yes"] = (model_df["children"] == "yes").astype(int)

    # Center a few continuous predictors to improve interpretability
    for col in ["age", "yearsmarried", "religiousness", "rating"]:
        model_df[f"c_{col}"] = model_df[col] - model_df[col].mean()

    X = model_df[
        [
            "children_yes",
            "c_age",
            "c_yearsmarried",
            "c_religiousness",
            "c_rating",
            "education",
            "occupation",
        ]
    ]
    X = sm.add_constant(X)
    y = model_df["any_affair"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("Logistic regression: any_affair ~ children + controls")
    print(logit_model.summary())

    # Extract key effect for children
    children_coef = logit_model.params["children_yes"]
    children_se = logit_model.bse["children_yes"]
    children_p = logit_model.pvalues["children_yes"]

    print()
    print("Children coefficient (log-odds):", round(children_coef, 3))
    print("Std. error:", round(children_se, 3))
    print("p-value:", children_p)


if __name__ == "__main__":
    main()

