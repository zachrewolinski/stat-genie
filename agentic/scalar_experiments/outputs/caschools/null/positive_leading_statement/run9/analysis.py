import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: number of students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    print("Basic descriptives")
    print("------------------")
    print("N:", len(df))
    print("stratio mean:", df["stratio"].mean())
    print("stratio std:", df["stratio"].std())
    print("avgscore mean:", df["avgscore"].mean())
    print("avgscore std:", df["avgscore"].std())
    print()

    def corr_and_reg(y_col: str):
        x = df["stratio"]
        y = df[y_col]
        r, p = stats.pearsonr(x, y)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return r, p, model

    for score_col in ["read", "math", "avgscore"]:
        r, p, model = corr_and_reg(score_col)
        print(f"=== {score_col} vs stratio ===")
        print("Pearson r:", round(r, 3), "p:", f"{p:.4g}")
        print(
            "OLS slope (per +1 student/teacher):",
            round(model.params["stratio"], 3),
            "p:",
            f"{model.pvalues['stratio']:.4g}",
            "R^2:",
            round(model.rsquared, 3),
        )
        print()

    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer", "students"]
    X = df[["stratio"] + controls]
    X = sm.add_constant(X)
    y = df["avgscore"]
    model_full = sm.OLS(y, X).fit()

    print("=== avgscore on stratio + controls ===")
    print(
        "coef stratio:",
        round(model_full.params["stratio"], 3),
        "p:",
        f"{model_full.pvalues['stratio']:.4g}",
    )
    print("R^2 full model:", round(model_full.rsquared, 3))
    print()


if __name__ == "__main__":
    main()

