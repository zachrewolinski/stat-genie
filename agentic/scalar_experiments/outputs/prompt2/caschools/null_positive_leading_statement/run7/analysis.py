import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: higher values = more students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance as the average of reading and math scores
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with any missing values in variables used below (defensive, though dataset is clean)
    df_model = df[
        [
            "avgscore",
            "stratio",
            "income",
            "lunch",
            "calworks",
            "english",
            "expenditure",
            "students",
        ]
    ].dropna()

    print("Number of observations:", len(df_model))
    print()

    # Simple Pearson correlation between student-teacher ratio and average score
    corr = df_model["avgscore"].corr(df_model["stratio"])
    print("Correlation between avgscore and stratio:", corr)
    print()

    # Bivariate regression: avgscore ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["avgscore"], X_simple).fit()
    print("Bivariate regression: avgscore ~ stratio")
    print(model_simple.summary())
    print()

    # Multivariate regression with demographic and resource controls
    X_controls = df_model[
        [
            "stratio",
            "income",
            "lunch",
            "calworks",
            "english",
            "expenditure",
            "students",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["avgscore"], X_controls).fit()
    print("Multivariate regression: avgscore ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

