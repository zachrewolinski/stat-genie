import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    print("Basic description of student–teacher ratio and scores")
    print(df[["stratio", "avg_score"]].describe(), end="\n\n")

    # Simple bivariate association
    corr = df["stratio"].corr(df["avg_score"])
    print(f"Pearson correlation (stratio vs avg_score): {corr:.4f}")

    x_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avg_score"], x_simple).fit()

    print("\nSimple OLS: avg_score ~ stratio")
    print(model_simple.summary())

    # Multivariable model controlling for key demographics/resources
    covariates = ["stratio", "income", "calworks", "lunch", "english", "expenditure"]
    x_multi = sm.add_constant(df[covariates])
    model_multi = sm.OLS(df["avg_score"], x_multi).fit()

    print("\nMultivariable OLS: avg_score ~ stratio + controls")
    print(model_multi.summary())

    coef_stratio_simple = model_simple.params["stratio"]
    p_stratio_simple = model_simple.pvalues["stratio"]
    coef_stratio_multi = model_multi.params["stratio"]
    p_stratio_multi = model_multi.pvalues["stratio"]

    print("\nKey results:")
    print(
        f"Simple model: coef(stratio) = {coef_stratio_simple:.4f}, "
        f"p = {p_stratio_simple:.4g}"
    )
    print(
        f"Multivariable model: coef(stratio) = {coef_stratio_multi:.4f}, "
        f"p = {p_stratio_multi:.4g}"
    )


if __name__ == "__main__":
    main()

