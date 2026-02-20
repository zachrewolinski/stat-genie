import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    summary = {
        "n": int(df.shape[0]),
        "stratio_desc": df["stratio"].describe().to_dict(),
        "testscr_desc": df["testscr"].describe().to_dict(),
        "corr_stratio_testscr": float(df[["stratio", "testscr"]].corr().iloc[0, 1]),
    }

    print("Sample size:", summary["n"])
    print("\nStudent–teacher ratio (stratio) summary:")
    for k, v in summary["stratio_desc"].items():
        print(f"  {k}: {v}")

    print("\nAverage test score (testscr) summary:")
    for k, v in summary["testscr_desc"].items():
        print(f"  {k}: {v}")

    print("\nCorrelation between stratio and testscr:")
    print(summary["corr_stratio_testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary())

    # Multiple regression with standard covariates used in caschools analyses
    covariates = ["stratio", "income", "english", "lunch", "calworks"]
    X_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()
    print("\nMultiple OLS: testscr ~ stratio + income + english + lunch + calworks")
    print(model_multi.summary())


if __name__ == "__main__":
    main()

