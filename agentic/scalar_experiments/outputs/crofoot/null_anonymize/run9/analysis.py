import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Rename key columns for clarity
    df = df.rename(
        columns={
            "feature4": "win",  # 1 if focal group won
            "feature5": "dist_focal",  # distance of focal group from its home-range center
            "feature6": "dist_other",  # distance of other group from its home-range center
            "feature7": "n_focal",  # number of individuals in focal group
            "feature8": "n_other",  # number of individuals in other group
        }
    )

    # Construct key predictors reflecting the research question
    df["rel_group_size"] = df["n_focal"] - df["n_other"]
    df["home_advantage"] = df["dist_other"] - df["dist_focal"]
    # Positive home_advantage means the focal group is closer to its home-range center

    print("Basic descriptives (n = {} contests)".format(len(df)))
    print(df[["win", "rel_group_size", "home_advantage"]].describe())

    # Correlations for quick inspection
    print("\nPairwise correlations:")
    print(df[["win", "rel_group_size", "home_advantage"]].corr())

    # Logistic regression: probability focal group wins as a function of
    # relative group size and home-range advantage.
    model = smf.logit("win ~ rel_group_size + home_advantage", data=df).fit(disp=False)

    print("\nLogistic regression summary:")
    print(model.summary2())
    print("\nCoefficients:")
    print(model.params)
    print("\nP-values:")
    print(model.pvalues)


if __name__ == "__main__":
    main()

