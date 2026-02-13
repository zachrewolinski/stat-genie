import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df["tooth_type"] = df["sockets"]

    mismatches_genus_missing = (df["genus"] > df["age"]).sum()
    mismatches_age_missing = (df["age"] > df["genus"]).sum()

    print(
        "Rows with num_missing > n_sockets if "
        "genus=num_missing, age=n_sockets:",
        mismatches_genus_missing,
    )
    print(
        "Rows with num_missing > n_sockets if "
        "age=num_missing, genus=n_sockets:",
        mismatches_age_missing,
    )

    if mismatches_genus_missing <= mismatches_age_missing:
        print("Using mapping: genus = num_missing, age = n_sockets.")
        df["num_missing"] = df["genus"]
        df["n_sockets"] = df["age"]
    else:
        print("Using mapping: age = num_missing, genus = n_sockets.")
        df["num_missing"] = df["age"]
        df["n_sockets"] = df["genus"]

    df["age_at_death"] = df["pop"]
    df["age_uncertainty"] = df["num_amtl"]
    df["prob_male"] = df["stdev_age"]
    df["genus_label"] = df["tooth_class"]

    df = df[df["n_sockets"] > 0].copy()
    df["prop_missing"] = df["num_missing"] / df["n_sockets"]

    print("Genus counts:")
    print(df["genus_label"].value_counts())
    print("\nTooth type counts:")
    print(df["tooth_type"].value_counts())

    print("\nSummary of prop_missing by genus:")
    print(df.groupby("genus_label")["prop_missing"].describe())

    formula = (
        "prop_missing ~ C(genus_label, Treatment(reference='Homo sapiens'))"
        " + age_at_death + prob_male + C(tooth_type)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    genus_params = model.params.filter(like="genus_label")
    genus_pvalues = model.pvalues.filter(like="genus_label")

    print("\nModel coefficients for genus_label (log-odds vs Homo sapiens):")
    print(genus_params)

    print("\nP-values for genus effects:")
    print(genus_pvalues)

    genus_levels = sorted(df["genus_label"].unique())
    marginal_means: dict[str, float] = {}

    for g in genus_levels:
        df_g = df.copy()
        df_g["genus_label"] = g
        pred = model.predict(df_g)
        marginal_means[g] = float(pred.mean())

    print("\nMarginal predicted AMTL rates by genus (controlling for covariates):")
    for g, v in marginal_means.items():
        print(f"{g}: {v:.4f}")

    homo_rate = marginal_means.get("Homo sapiens")
    nonhuman_rates = [v for g, v in marginal_means.items() if g != "Homo sapiens"]
    avg_nonhuman = float(np.mean(nonhuman_rates))
    diff = homo_rate - avg_nonhuman

    print(f"\nHomo sapiens marginal AMTL rate: {homo_rate:.4f}")
    print(f"Average non-human marginal AMTL rate: {avg_nonhuman:.4f}")
    print(f"Difference (Homo - non-human): {diff:.4f}")

    print("\nFitting Poisson model with log(n_sockets) offset...")

    poisson_formula = (
        "num_missing ~ C(genus_label, Treatment(reference='Homo sapiens'))"
        " + age_at_death + prob_male + C(tooth_type)"
    )

    poisson_model = smf.glm(
        formula=poisson_formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["n_sockets"]),
    ).fit()

    poisson_genus_params = poisson_model.params.filter(like="genus_label")
    poisson_genus_pvalues = poisson_model.pvalues.filter(like="genus_label")

    print(
        "\nPoisson model genus coefficients "
        "(log rate ratio vs Homo sapiens):"
    )
    print(poisson_genus_params)

    print("\nPoisson model genus p-values:")
    print(poisson_genus_pvalues)

    marginal_means_poisson: dict[str, float] = {}

    for g in genus_levels:
        df_g = df.copy()
        df_g["genus_label"] = g
        pred_counts = poisson_model.predict(
            df_g, offset=np.log(df_g["n_sockets"])
        )
        rate = (pred_counts / df_g["n_sockets"]).mean()
        marginal_means_poisson[g] = float(rate)

    print("\nPoisson-based marginal AMTL rates by genus:")
    for g, v in marginal_means_poisson.items():
        print(f"{g}: {v:.4f}")

    homo_rate_p = marginal_means_poisson.get("Homo sapiens")
    nonhuman_rates_p = [
        v for g, v in marginal_means_poisson.items() if g != "Homo sapiens"
    ]
    avg_nonhuman_p = float(np.mean(nonhuman_rates_p))
    diff_p = homo_rate_p - avg_nonhuman_p

    print(f"\nHomo sapiens Poisson-based rate: {homo_rate_p:.4f}")
    print(f"Average non-human Poisson-based rate: {avg_nonhuman_p:.4f}")
    print(f"Difference (Homo - non-human, Poisson): {diff_p:.4f}")


if __name__ == "__main__":
    main()
