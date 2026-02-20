import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Restrict to humans and the three non-human genera of interest
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Basic cleaning: require valid counts and sockets > 0
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class"])
    df = df[df["sockets"] > 0].copy()

    # Binary indicator for modern humans
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in this tooth class
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Summarize raw AMTL frequencies by genus
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(prop_missing=lambda x: x["num_amtl"] / x["sockets"])
    )
    print("Raw AMTL frequencies by genus:")
    print(genus_summary)
    print()

    # Binomial regression: proportion missing with binomial family and socket counts as weights
    formula = "prop_missing ~ human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())
    print()

    coef_human = result.params["human"]
    se_human = result.bse["human"]
    pval_human = result.pvalues["human"]
    or_human = float(np.exp(coef_human))

    print(f"Coefficient for human (log-odds): {coef_human:.4f}")
    print(f"Std. error: {se_human:.4f}")
    print(f"Odds ratio (humans vs non-humans): {or_human:.3f}")
    print(f"P-value: {pval_human:.4g}")


if __name__ == "__main__":
    main()

