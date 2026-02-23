import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    df["focal_win"] = df["feature4"]

    # Relative group size: positive when focal group is larger
    df["rel_group_size"] = df["feature7"] - df["feature8"]

    # Contest location: 1 if focal group is closer to its home-range center
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    print("N rows:", len(df))
    print()

    # Descriptive summaries for main predictors
    print("Win rate by relative group size (focal larger vs not):")
    larger_mask = df["rel_group_size"] > 0
    not_larger_mask = ~larger_mask
    print(
        "Focal larger (rel_group_size > 0): "
        f"{df.loc[larger_mask, 'focal_win'].mean():.3f} "
        f"(n={larger_mask.sum()})"
    )
    print(
        "Focal not larger (<= 0): "
        f"{df.loc[not_larger_mask, 'focal_win'].mean():.3f} "
        f"(n={not_larger_mask.sum()})"
    )
    print()

    print("Win rate by contest location (focal closer vs not):")
    closer_mask = df["focal_closer"] == 1
    not_closer_mask = ~closer_mask
    print(
        "Focal closer: "
        f"{df.loc[closer_mask, 'focal_win'].mean():.3f} "
        f"(n={closer_mask.sum()})"
    )
    print(
        "Focal not closer: "
        f"{df.loc[not_closer_mask, 'focal_win'].mean():.3f} "
        f"(n={not_closer_mask.sum()})"
    )
    print()

    print("rel_group_size summary:")
    print(df["rel_group_size"].describe())
    print()

    print("focal_closer value counts:")
    print(df["focal_closer"].value_counts())
    print()

    # Fit logistic regression: probability focal group wins
    formula = "focal_win ~ rel_group_size + focal_closer"
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    result = model.fit()

    print(result.summary())

    params = result.params
    conf_int = result.conf_int()
    odds_ratios = np.exp(params)
    or_ci_lower = np.exp(conf_int[0])
    or_ci_upper = np.exp(conf_int[1])

    print("\nOdds ratios with 95% CI:")
    for term in params.index:
        print(
            f"{term:15s} OR={odds_ratios[term]:.3f} "
            f"95% CI=({or_ci_lower[term]:.3f}, {or_ci_upper[term]:.3f}) "
            f"p={result.pvalues[term]:.3f}"
        )


if __name__ == "__main__":
    main()
