import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: number of students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    print("Summary statistics")
    print("------------------")
    print("Student-teacher ratio (stratio):")
    print(df["stratio"].describe())
    print("\nAverage test score (avgscore):")
    print(df["avgscore"].describe())

    print("\nPearson correlations between stratio and test scores")
    for col in ["read", "math", "avgscore"]:
        r, p = stats.pearsonr(df["stratio"], df[col])
        print(f"stratio vs {col}: r = {r:.3f}, p-value = {p:.3g}")

    # OLS regression: avgscore ~ stratio
    y = df["avgscore"]
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nOLS regression: avgscore ~ stratio")
    print(model_simple.summary())

    # OLS regression with demographic and resource controls
    controls = ["income", "english", "lunch", "calworks"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()
    print("\nOLS regression: avgscore ~ stratio + income + english + lunch + calworks")
    print(model_controls.summary())

    # Quartile analysis of stratio
    df["stratio_quartile"] = pd.qcut(df["stratio"], 4, labels=[1, 2, 3, 4])
    group_means = df.groupby("stratio_quartile")["avgscore"].mean()
    print("\nAverage test score by stratio quartile (1 = lowest stratio, 4 = highest stratio):")
    print(group_means)


if __name__ == "__main__":
    main()

