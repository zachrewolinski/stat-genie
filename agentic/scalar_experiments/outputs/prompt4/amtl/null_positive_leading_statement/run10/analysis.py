import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Binary indicator for modern humans
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Simple descriptive rates by genus
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(rate=lambda x: x["num_amtl"] / x["sockets"])
    )

    print("AMTL rates by genus (num_amtl / sockets):")
    print(genus_summary)
    print()

    # Binomial regression on proportion of AMTL using sockets as weights
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    formula = "prop_amtl ~ human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial GLM results:")
    print(result.summary())
    print()

    human_coef = result.params.get("human", np.nan)
    human_se = result.bse.get("human", np.nan)
    human_p = result.pvalues.get("human", np.nan)
    human_or = float(np.exp(human_coef)) if np.isfinite(human_coef) else np.nan

    print("Human indicator coefficient (log-odds):", human_coef)
    print("Standard error:", human_se)
    print("p-value:", human_p)
    print("Odds ratio (exp(coef)):", human_or)


if __name__ == "__main__":
    main()

