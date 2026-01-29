import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: keep rows with valid counts and covariates
    df = df.copy()
    df = df[df["sockets"].notna() & df["num_amtl"].notna()]
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]
    df = df[df["age"].notna() & df["prob_male"].notna() & df["tooth_class"].notna() & df["genus"].notna()]

    # Create human indicator
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # AMTL rate for binomial regression
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Model with controls for age, sex (prob_male), and tooth class
    formula = "amtl_rate ~ human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    result = model.fit()

    # Extract human coefficient
    coef = result.params.get("human", np.nan)
    pval = result.pvalues.get("human", np.nan)

    # Determine conclusion
    if np.isfinite(coef) and np.isfinite(pval) and (coef > 0) and (pval < 0.05):
        answer = "Yes"
        reason = (
            "After adjusting for age, sex, and tooth class, the human indicator has a positive and statistically "
            "significant association with AMTL frequency."
        )
    else:
        answer = "No"
        reason = (
            "After adjusting for age, sex, and tooth class, the evidence does not show a significantly higher AMTL "
            "frequency for humans compared with non-human primates."
        )

    with open("conclusion.txt", "w") as f:
        f.write(answer + "\n")
        f.write(reason + "\n")


if __name__ == "__main__":
    main()
