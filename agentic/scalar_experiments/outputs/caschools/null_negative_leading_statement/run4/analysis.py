import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Key derived variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Number of districts:", len(df))
    print("Student-teacher ratio summary:")
    print(df["stratio"].describe())
    print("\nTest score summary:")
    print(df["testscr"].describe())

    # Bivariate relationship
    corr = df["stratio"].corr(df["testscr"])
    print("\nPearson correlation between stratio and testscr:", corr)

    # Simple OLS: testscr ~ stratio
    X1 = sm.add_constant(df["stratio"])
    model1 = sm.OLS(df["testscr"], X1).fit()
    print("\n=== Simple OLS: testscr ~ stratio ===")
    print(model1.summary())

    # Multiple OLS controlling for key demographics
    controls = ["income", "english", "lunch"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(df["testscr"], X2).fit()
    print("\n=== Multiple OLS: testscr ~ stratio + income + english + lunch ===")
    print(model2.summary())


if __name__ == "__main__":
    main()

