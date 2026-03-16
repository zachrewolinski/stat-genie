import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    key_cols = [
        "stratio",
        "testscr",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df.dropna(subset=key_cols).copy()

    print("Sample size:", len(df_model))
    print("\nStudent-teacher ratio (stratio) summary:")
    print(df_model["stratio"].describe())

    print("\nTest score (testscr) summary:")
    print(df_model["testscr"].describe())

    r, p = stats.pearsonr(df_model["stratio"], df_model["testscr"])
    print("\nPearson correlation between stratio and testscr:")
    print("r =", r, "p-value =", p)

    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    X_controls = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit(cov_type="HC1")
    print("\nMultiple OLS with controls (robust SE, HC1):")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

