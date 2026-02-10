import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: higher values = larger classes.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obviously invalid rows (e.g., missing values).
    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate relationship.
    corr = df["stratio"].corr(df["testscr"])

    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()
    coef = model.params["stratio"]
    pval = model.pvalues["stratio"]
    r2 = model.rsquared

    print("Number of districts:", len(df))
    print("Correlation (stratio, testscr):", corr)
    print("OLS coef on stratio:", coef)
    print("p-value for stratio:", pval)
    print("R-squared:", r2)

    # Also fit a model with basic controls to see if the sign is robust.
    controls = ["income", "english", "calworks", "lunch", "expenditure"]
    available_controls = [c for c in controls if c in df.columns]
    if available_controls:
        Xc = sm.add_constant(df[["stratio"] + available_controls])
        model_c = sm.OLS(df["testscr"], Xc).fit()
        print("\nWith controls:")
        print("OLS coef on stratio (controls):", model_c.params["stratio"])
        print("p-value for stratio (controls):", model_c.pvalues["stratio"])
        print("R-squared with controls:", model_c.rsquared)


if __name__ == "__main__":
    main()

