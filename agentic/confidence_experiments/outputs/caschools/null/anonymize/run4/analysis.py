import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct variables based on metadata in info.json
    # Student-teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avgscore"] = (df["feature14"] + df["feature15"]) / 2.0

    # Simple correlation between class size (student-teacher ratio) and performance
    corr = df["stratio"].corr(df["avgscore"])

    # Bivariate OLS: avgscore ~ stratio
    X1 = sm.add_constant(df["stratio"])
    y = df["avgscore"]
    model1 = sm.OLS(y, X1).fit()

    # Multivariate OLS with key demographic and resource controls
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    X2 = sm.add_constant(df[["stratio"] + controls])
    model2 = sm.OLS(y, X2).fit()

    coef1 = model1.params["stratio"]
    pval1 = model1.pvalues["stratio"]

    coef2 = model2.params["stratio"]
    pval2 = model2.pvalues["stratio"]

    print("Simple correlation (stratio vs avgscore):", corr)
    print("Bivariate OLS: coef(stratio) =", coef1, "p-value =", pval1)
    print("Multivariate OLS with controls: coef(stratio) =", coef2, "p-value =", pval2)


if __name__ == "__main__":
    main()

