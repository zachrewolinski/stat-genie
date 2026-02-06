import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def main():
    df = pd.read_csv("panda_nuts.csv")
    # Efficiency: nuts opened per second
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    # Basic OLS with categorical sex and help
    model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")
    print(model.summary())

    # ANOVA on non-robust model for overall effects (type II)
    model_nr = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()
    anova = anova_lm(model_nr, typ=2)
    print("\nANOVA (Type II):")
    print(anova)

    # Group means for context
    print("\nGroup means:")
    print(df.groupby("sex")["efficiency"].mean())
    print(df.groupby("help")["efficiency"].mean())

    # p-values from robust model
    print("\nRobust p-values:")
    print(model.pvalues)


if __name__ == "__main__":
    main()
