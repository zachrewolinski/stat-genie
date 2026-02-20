import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic grouped-binomial summary by genus
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(prop_amtl=lambda x: x["num_amtl"] / x["sockets"])
    )

    print("Overall AMTL by genus (num_amtl / sockets):")
    print(genus_summary)
    print()

    # Binomial regression: proportion of missing teeth with weights = sockets.
    # Include genus, age, sex estimate, and tooth class as predictors.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print(model.summary())
    print()

    # Extract genus coefficients (baseline is the first level alphabetically).
    # With the observed data this should be "Homo sapiens".
    params = model.params
    conf_int = model.conf_int()

    print("Genus effects relative to baseline (likely Homo sapiens):")
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus)[T.{genus}]"
        if term not in params.index:
            continue
        beta = params[term]
        ci_low, ci_high = conf_int.loc[term]
        # Effect of Homo sapiens vs genus g is -beta
        or_homo_vs_g = float(np.exp(-beta))
        or_low = float(np.exp(-ci_high))
        or_high = float(np.exp(-ci_low))
        print(
            f"Homo sapiens vs {genus}: "
            f"OR={or_homo_vs_g:.3f}, 95% CI [{or_low:.3f}, {or_high:.3f}]"
        )


if __name__ == "__main__":
    main()
