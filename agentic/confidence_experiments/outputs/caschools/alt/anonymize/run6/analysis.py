import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: mean of reading and math scores
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop rows with missing values in variables of interest, if any
    df_model = df[["avg_score", "stratio"]].dropna()

    # Simple correlation
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["avg_score"])

    # Simple OLS: avg_score ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["avg_score"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = [
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district avg income
        "feature13",  # % English learners
    ]
    cols = ["avg_score", "stratio"] + controls
    df_m = df[cols].dropna()
    X_full = sm.add_constant(df_m[["stratio"] + controls])
    model_full = sm.OLS(df_m["avg_score"], X_full).fit()

    print("N (simple):", len(df_model))
    print("Pearson r (stratio, avg_score):", r)
    print("Pearson p-value:", p_corr)
    print()
    print("Simple OLS: avg_score ~ stratio")
    print(model_simple.summary())
    print()
    print("Multiple OLS with controls:")
    print(model_full.summary())


if __name__ == "__main__":
    main()

