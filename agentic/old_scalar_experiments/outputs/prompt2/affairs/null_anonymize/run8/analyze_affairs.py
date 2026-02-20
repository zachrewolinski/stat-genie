import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affairs in the past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Ensure categorical variables are treated as such
    df["feature3"] = df["feature3"].astype("category")  # gender
    df["feature6"] = df["feature6"].astype("category")  # children

    # Descriptive statistics by children status
    desc = (
        df.groupby("feature6")
        .agg(
            mean_affairs=("feature2", "mean"),
            prop_any=("any_affair", "mean"),
            count=("any_affair", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status (feature6):")
    print(desc.to_string(index=False))
    print()

    # Logistic regression for having any affair, controlling for covariates
    formula = (
        "any_affair ~ C(feature6) + C(feature3) + "
        "feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )

    model = smf.logit(formula, data=df).fit(disp=False)

    print("Logistic regression results for any_affair:")
    print(model.summary())
    print()

    # Extract effect of having children (yes vs no)
    coef_children = model.params.get("C(feature6)[T.yes]")
    pval_children = model.pvalues.get("C(feature6)[T.yes]")
    if coef_children is not None:
        odds_ratio_children = float(np.exp(coef_children))
    else:
        odds_ratio_children = np.nan

    print("Effect of having children on any_affair (yes vs no):")
    print(f"  Log-odds coefficient: {coef_children}")
    print(f"  Odds ratio: {odds_ratio_children}")
    print(f"  p-value: {pval_children}")


if __name__ == "__main__":
    main()

