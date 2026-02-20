import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2

    cols = [
        "avg_score",
        "stratio",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    df = df[cols].dropna()

    corr = df["stratio"].corr(df["avg_score"])
    print(f"Correlation between student-teacher ratio and avg_score: {corr:.4f}")

    X1 = sm.add_constant(df[["stratio"]])
    model1 = sm.OLS(df["avg_score"], X1).fit()
    coef1 = model1.params["stratio"]
    pval1 = model1.pvalues["stratio"]
    print("Model 1: avg_score ~ stratio")
    print(f"  Coef(stratio): {coef1:.4f}, p-value: {pval1:.4g}")

    controls = [
        "stratio",
        "income",
        "english",
        "calworks",
        "lunch",
        "computer",
        "expenditure",
    ]
    X2 = sm.add_constant(df[controls])
    model2 = sm.OLS(df["avg_score"], X2).fit()
    coef2 = model2.params["stratio"]
    pval2 = model2.pvalues["stratio"]
    print("Model 2: avg_score ~ stratio + controls")
    print(f"  Coef(stratio): {coef2:.4f}, p-value: {pval2:.4g}")


if __name__ == "__main__":
    main()

