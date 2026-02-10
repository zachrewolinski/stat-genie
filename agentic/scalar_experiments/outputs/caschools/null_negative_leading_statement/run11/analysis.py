import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Drop any obvious missing values if present
    df_model = df.dropna(
        subset=[
            "avgscore",
            "stratio",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
        ]
    )

    # Simple bivariate regression: avgscore on student–teacher ratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["avgscore"], X_simple).fit()

    # Multiple regression with key covariates to adjust for demographics/resources
    X_controls = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["avgscore"], X_controls).fit()

    # Print key results for inspection in the shell
    print("Simple regression: avgscore ~ stratio")
    print(model_simple.summary())
    print("\n\nMultiple regression with controls:")
    print(model_controls.summary())

    # Also print basic Pearson correlation
    corr = df["stratio"].corr(df["avgscore"])
    print(f"\nPearson correlation between stratio and avgscore: {corr:.4f}")


if __name__ == "__main__":
    main()

