import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio
    df["str"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Keep relevant columns and drop missing
    cols = ["str", "score", "read", "math", "lunch", "income", "english"]
    d = df[cols].dropna()

    corr_score = d[["str", "score"]].corr().iloc[0, 1]
    corr_read = d[["str", "read"]].corr().iloc[0, 1]
    corr_math = d[["str", "math"]].corr().iloc[0, 1]

    # Simple regression: score ~ str
    X_simple = sm.add_constant(d["str"])
    model_simple = sm.OLS(d["score"], X_simple).fit()

    # Controlled regression: score ~ str + lunch + income + english
    X_ctrl = sm.add_constant(d[["str", "lunch", "income", "english"]])
    model_ctrl = sm.OLS(d["score"], X_ctrl).fit()

    print("Correlation (STR vs score):", corr_score)
    print("Correlation (STR vs read):", corr_read)
    print("Correlation (STR vs math):", corr_math)
    print("Simple regression coef (str):", model_simple.params["str"])
    print("Simple regression p-value (str):", model_simple.pvalues["str"])
    print("Controlled regression coef (str):", model_ctrl.params["str"])
    print("Controlled regression p-value (str):", model_ctrl.pvalues["str"])


if __name__ == "__main__":
    main()
