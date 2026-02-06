import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("hurricane.csv")

    # Basic summaries
    df["log_deaths"] = np.log1p(df["alldeaths"])

    # Correlation between name femininity and deaths
    corr = df["masfem"].corr(df["alldeaths"])

    # OLS on log deaths with controls (robust SE)
    X = df[["masfem", "wind", "min", "category", "ndam15", "year"]].copy()
    X = sm.add_constant(X)
    ols = sm.OLS(df["log_deaths"], X).fit(cov_type="HC3")

    # Alternative specification using binary gender
    X2 = df[["gender_mf", "wind", "min", "category", "ndam15", "year"]].copy()
    X2 = sm.add_constant(X2)
    ols_gender = sm.OLS(df["log_deaths"], X2).fit(cov_type="HC3")

    # Count model (negative binomial GLM)
    nb = smf.glm(
        formula="alldeaths ~ masfem + wind + min + category + ndam15 + year",
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()

    print("Rows:", len(df))
    print("Deaths summary:\n", df["alldeaths"].describe())
    print("Masfem summary:\n", df["masfem"].describe())
    print("Correlation (masfem, alldeaths):", corr)

    print("\nOLS log1p(deaths) ~ masfem + controls (HC3 SE)")
    print(ols.summary())

    print("\nOLS log1p(deaths) ~ gender_mf + controls (HC3 SE)")
    print(ols_gender.summary())

    print("\nNegative Binomial GLM deaths ~ masfem + controls")
    print(nb.summary())


if __name__ == "__main__":
    main()
