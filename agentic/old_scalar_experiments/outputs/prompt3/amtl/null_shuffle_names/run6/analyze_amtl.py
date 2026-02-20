import pandas as pd
import statsmodels.api as sm
import numpy as np


def main() -> None:
    # Load the dataset
    df = pd.read_csv("amtl.csv")

    # Rename columns to more descriptive names based on metadata
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior/Posterior/Premolar
            "prob_male": "specimen_id",
            "genus": "num_missing",  # number of missing teeth (AMTL count)
            "age": "num_sockets",  # observable sockets
            "pop": "age_at_death",  # estimated age at death
            "num_amtl": "age_uncertainty",
            "stdev_age": "prob_male",  # numeric estimate of sex (probability male)
            "tooth_class": "genus",  # Homo sapiens, Pan, Papio, Pongo
            "specimen": "region",
        }
    )

    # Keep only the genera of interest
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])].copy()

    # Basic data quality filters
    df = df[
        (df["num_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["num_sockets"])
    ].copy()

    # Create variables needed for modeling
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["missing_prop"] = df["num_missing"] / df["num_sockets"]

    # Fit a binomial GLM on aggregated data:
    #   logit(Pr[AMTL]) ~ human vs non-human + age + sex + tooth class
    formula = "missing_prop ~ is_human + age_at_death + prob_male + C(tooth_class)"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    print(result.summary())
    print()

    # Weighted mean missing proportion by genus
    group_means = (
        df.groupby("genus")
        .apply(
            lambda g: (g["missing_prop"] * g["num_sockets"]).sum()
            / g["num_sockets"].sum()
        )
        .sort_values()
    )
    print("Weighted missing proportion by genus:")
    print(group_means)

    coef = result.params["is_human"]
    pval = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    print(f"\nHuman indicator coef (log-odds): {coef:.4f}")
    print(f"Human indicator odds ratio: {odds_ratio:.3f}")
    print(f"Human indicator p-value: {pval:.4g}")


if __name__ == "__main__":
    main()
