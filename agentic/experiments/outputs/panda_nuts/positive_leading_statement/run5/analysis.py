import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main():
    df = pd.read_csv("panda_nuts.csv")

    # Basic rate metric
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    # OLS on efficiency (nuts per second)
    ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

    # Poisson rate model with offset for time
    df["log_seconds"] = np.log(df["seconds"])
    poisson_model = smf.glm(
        "nuts_opened ~ age + C(sex) + C(help)",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_seconds"],
    ).fit()

    print("OLS efficiency model (robust SE):")
    print(ols_model.summary())
    print("\nPoisson rate model (offset log(seconds)):")
    print(poisson_model.summary())

    # Collect key p-values for inference
    pvals_ols = ols_model.pvalues
    pvals_pois = poisson_model.pvalues
    print("\nKey p-values (OLS):")
    print(pvals_ols[["age", "C(sex)[T.m]", "C(help)[T.y]"]])
    print("\nKey p-values (Poisson):")
    print(pvals_pois[["age", "C(sex)[T.m]", "C(help)[T.y]"]])


if __name__ == "__main__":
    main()
