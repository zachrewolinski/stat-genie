import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio and combined test score
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "testscr", "income", "english", "lunch"])

    # Correlation between student-teacher ratio and test scores
    r, p = stats.pearsonr(df["stratio"], df["testscr"])

    # Simple linear regression: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()

    # Multiple regression with key socioeconomic controls
    X2 = df[["stratio", "income", "english", "lunch"]]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df["testscr"], X2).fit()

    print("N observations:", len(df))
    print("Pearson correlation (testscr vs stratio):")
    print(f"  r = {r:.3f}, p-value = {p:.3g}")
    print()
    print("Model 1: testscr ~ stratio")
    print(model1.summary())
    print()
    print("Model 2: testscr ~ stratio + income + english + lunch")
    print(model2.summary())


if __name__ == "__main__":
    main()

