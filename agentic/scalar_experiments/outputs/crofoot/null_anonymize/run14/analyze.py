import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["feature4"]

    # Relative group size and location metrics (differences)
    df_rel = pd.DataFrame(
        {
            # Positive if focal group is larger
            "diff_total": df["feature7"] - df["feature8"],
            "diff_males": df["feature9"] - df["feature10"],
            "diff_females": df["feature11"] - df["feature12"],
            # Positive if focal group is farther from its home-range center
            "diff_dist_center": df["feature5"] - df["feature6"],
        }
    )

    X = sm.add_constant(df_rel)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logistic regression of winning on relative size and location")
    print(result.summary())
    print("\nCoefficients and p-values (difference metrics):")
    print(pd.DataFrame({"coef": result.params, "pvalue": result.pvalues}))

    # Alternative parameterisation: size ratio and simple home-advantage indicator
    df_alt = pd.DataFrame(
        {
            "size_ratio": df["feature7"] / df["feature8"],
            "home_advantage": (df["feature5"] < df["feature6"]).astype(int),
        }
    )
    X_alt = sm.add_constant(df_alt)
    model_alt = sm.Logit(y, X_alt)
    result_alt = model_alt.fit(disp=False)

    print(
        "\nLogistic regression with size ratio and home-advantage indicator"
    )
    print(result_alt.summary())
    print("\nCoefficients and p-values (ratio/home-advantage metrics):")
    print(pd.DataFrame({"coef": result_alt.params, "pvalue": result_alt.pvalues}))


if __name__ == "__main__":
    main()
