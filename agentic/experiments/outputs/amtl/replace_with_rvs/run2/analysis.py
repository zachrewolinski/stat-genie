import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Binary indicator for modern humans vs non-human primates
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Define total observed teeth as missing + observable sockets
    df["total_teeth"] = df["num_amtl"] + df["sockets"]
    df["failures"] = df["total_teeth"] - df["num_amtl"]

    # Binomial GLM with grouped counts
    exog = patsy.dmatrix(
        "human + age + prob_male + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )
    endog = df[["num_amtl", "failures"]]

    model = sm.GLM(endog, exog, family=sm.families.Binomial())

    # Cluster-robust SE by specimen to account for multiple rows per individual
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    human_coef = result.params["human"]
    human_p = result.pvalues["human"]
    human_or = float(np.exp(human_coef))

    print("Human coefficient (log-odds):", human_coef)
    print("Human odds ratio:", human_or)
    print("Human p-value:", human_p)


if __name__ == "__main__":
    main()
