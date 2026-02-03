import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("hurricane.csv")

    # Basic derived variables for skewed outcomes
    df["log_deaths"] = np.log1p(df["alldeaths"])
    df["log_ndam15"] = np.log1p(df["ndam15"])

    print("Rows:", len(df))
    print("Deaths summary:", df["alldeaths"].describe())
    print("Femininity (masfem) summary:", df["masfem"].describe())
    print()

    # Simple correlation between femininity and deaths
    corr = df[["masfem", "alldeaths"]].corr().iloc[0, 1]
    corr_log = df[["masfem", "log_deaths"]].corr().iloc[0, 1]
    print(f"Correlation masfem vs alldeaths: {corr:.3f}")
    print(f"Correlation masfem vs log_deaths: {corr_log:.3f}")
    print()

    # Regression controlling for hurricane intensity and time
    # Using wind, minimum pressure, category, damage, and elapsed years
    formula = (
        "log_deaths ~ masfem + wind + min + category + log_ndam15 + elapsedyrs"
    )
    model = smf.ols(formula, data=df).fit()
    print("OLS on log_deaths with controls")
    print(model.summary())
    print()

    # Binary gender indicator model
    formula_gender = (
        "log_deaths ~ gender_mf + wind + min + category + log_ndam15 + elapsedyrs"
    )
    model_gender = smf.ols(formula_gender, data=df).fit()
    print("OLS on log_deaths with gender_mf")
    print(model_gender.summary())
    print()

    # Group comparison (female vs male names)
    group_stats = df.groupby("gender_mf")["log_deaths"].agg(["mean", "std", "count"])
    print("Group stats for log_deaths by gender_mf (0=male,1=female):")
    print(group_stats)
    print()

    # Two-sample t-test on log_deaths by gender
    male = df[df["gender_mf"] == 0]["log_deaths"]
    female = df[df["gender_mf"] == 1]["log_deaths"]
    tstat, pval, dfree = sm.stats.ttest_ind(male, female, usevar="unequal")
    print(f"T-test log_deaths female vs male: t={tstat:.3f}, p={pval:.3f}")


if __name__ == "__main__":
    main()
