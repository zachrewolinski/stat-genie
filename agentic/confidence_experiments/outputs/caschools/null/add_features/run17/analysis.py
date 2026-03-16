import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math scores
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    df_model = df[["stratio", "avgscore", "calworks", "lunch", "english", "income", "expenditure"]].dropna()

    print("N (used in models):", len(df_model))
    print("Mean student-teacher ratio:", df_model["stratio"].mean())
    print("Std student-teacher ratio:", df_model["stratio"].std())
    print("Mean avg score:", df_model["avgscore"].mean())
    print("Std avg score:", df_model["avgscore"].std())

    # Simple correlation
    r, p = stats.pearsonr(df_model["stratio"], df_model["avgscore"])
    print("Pearson r (stratio vs avgscore):", r)
    print("Pearson p-value:", p)

    # Simple OLS: avgscore ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    simple_model = sm.OLS(df_model["avgscore"], X_simple).fit()
    print("\nSimple OLS: avgscore ~ stratio")
    print("coef_stratio:", simple_model.params["stratio"])
    print("pvalue_stratio:", simple_model.pvalues["stratio"])
    print("R2_simple:", simple_model.rsquared)

    # Multiple OLS with key controls
    controls = ["calworks", "lunch", "english", "income", "expenditure"]
    X_controls = sm.add_constant(df_model[["stratio"] + controls])
    multi_model = sm.OLS(df_model["avgscore"], X_controls).fit()
    print("\nMultiple OLS: avgscore ~ stratio + controls")
    print("coef_stratio_adj:", multi_model.params["stratio"])
    print("pvalue_stratio_adj:", multi_model.pvalues["stratio"])
    print("R2_multi:", multi_model.rsquared)


if __name__ == "__main__":
    main()

