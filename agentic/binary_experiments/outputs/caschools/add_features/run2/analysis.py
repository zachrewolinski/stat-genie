import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("caschools.csv")

    # Compute student-teacher ratio
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["score_avg"] = (df["read"] + df["math"]) / 2.0

    # Keep relevant columns and drop missing
    cols = ["stratio", "score_avg", "read", "math", "lunch", "english", "income"]
    df_sub = df[cols].dropna()

    # Simple correlation
    corr = df_sub["stratio"].corr(df_sub["score_avg"])

    # Simple OLS: score_avg ~ stratio
    X1 = sm.add_constant(df_sub[["stratio"]])
    model1 = sm.OLS(df_sub["score_avg"], X1).fit()

    # Controlled OLS: add key demographics and income
    X2 = sm.add_constant(df_sub[["stratio", "lunch", "english", "income"]])
    model2 = sm.OLS(df_sub["score_avg"], X2).fit()

    # Print concise results for inspection
    print("Correlation (stratio, score_avg):", corr)
    print("\nSimple OLS (score_avg ~ stratio):")
    print(model1.summary().tables[1])
    print("\nControlled OLS (score_avg ~ stratio + lunch + english + income):")
    print(model2.summary().tables[1])


if __name__ == "__main__":
    main()
