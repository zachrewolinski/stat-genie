import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = "amtl.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    # Create indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Response as proportion with binomial weights (sockets as trials)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract human effect
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    print(result.summary())
    print("\nHuman effect (log-odds):", coef)
    print("SE:", se)
    print("p-value:", pval)
    print("Odds ratio:", odds_ratio)

    # Save key stats for downstream use
    pd.DataFrame(
        {
            "coef": [coef],
            "se": [se],
            "pval": [pval],
            "odds_ratio": [odds_ratio],
        }
    ).to_csv("human_effect_summary.csv", index=False)

if __name__ == "__main__":
    main()
