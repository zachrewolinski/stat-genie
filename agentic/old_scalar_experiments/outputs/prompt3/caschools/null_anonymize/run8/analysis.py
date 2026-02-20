import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on metadata in info.json
    # Student-teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]
    # Academic performance: average of reading and math scores
    df["testscr"] = df[["feature14", "feature15"]].mean(axis=1)

    # Basic association: correlation
    corr = df["stratio"].corr(df["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df[["stratio"]])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression controlling for demographics and resources
    controls = ["feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(df["testscr"], X_multi).fit()

    # Print key results for inspection in the CLI
    print("Correlation between student-teacher ratio and test scores:", corr)
    print("\nSimple regression (testscr ~ stratio):")
    print("  Coefficient on stratio:", model_simple.params["stratio"])
    print("  p-value for stratio:", model_simple.pvalues["stratio"])
    print("  R-squared:", model_simple.rsquared)

    print("\nMultiple regression with controls:")
    print("  Coefficient on stratio:", model_multi.params["stratio"])
    print("  p-value for stratio:", model_multi.pvalues["stratio"])
    print("  95% CI for stratio:", model_multi.conf_int().loc["stratio"].to_dict())
    print("  R-squared:", model_multi.rsquared)


if __name__ == "__main__":
    main()

