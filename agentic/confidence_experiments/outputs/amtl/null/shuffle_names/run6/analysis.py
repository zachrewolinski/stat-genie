import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.contrasts import Treatment


def main() -> None:
    # Load the dataset
    df = pd.read_csv("amtl.csv")

    # Remap columns to clearer semantic names based on the metadata description
    df["tooth_class_cat"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]  # specimen identifier
    df["num_missing"] = df["genus"].astype(float)  # number of teeth missing (AMTL)
    df["n_sockets"] = df["age"].astype(float)  # number of observable sockets
    df["age_at_death"] = df["pop"].astype(float)  # estimated age at death
    df["age_uncertainty"] = df["num_amtl"].astype(float)  # uncertainty in age estimate
    df["prob_male_est"] = df["stdev_age"].astype(float)  # estimate of probability of being male
    df["genus_label"] = df["tooth_class"]  # genus: Homo sapiens, Pan, Papio, Pongo
    df["region"] = df["specimen"]  # population / region label

    print("Head with remapped columns:")
    print(
        df[
            [
                "genus_label",
                "tooth_class_cat",
                "age_at_death",
                "prob_male_est",
                "num_missing",
                "n_sockets",
            ]
        ].head()
    )

    # Basic sanity checks on counts
    invalid = (df["num_missing"] > df["n_sockets"]) | (df["num_missing"] < 0) | (
        df["n_sockets"] <= 0
    )
    print("Number of rows with invalid counts:", invalid.sum())

    df_valid = df.loc[~invalid].copy()

    # Proportion of missing teeth within the observable sockets
    df_valid["prop_missing"] = df_valid["num_missing"] / df_valid["n_sockets"]

    genera = df_valid["genus_label"].unique()
    print("Unique genera:", genera)

    # Use Homo sapiens as the reference genus when available
    if "Homo sapiens" in genera:
        ref_genus = "Homo sapiens"
    else:
        ref_genus = sorted(genera)[0]
    print("Reference genus for coding:", ref_genus)

    # Binomial regression with logit link using aggregated binomial data
    # Response: proportion missing, with n_sockets as frequency weights
    formula = (
        f"prop_missing ~ C(genus_label, Treatment(reference='{ref_genus}'))"
        " + age_at_death + prob_male_est + C(tooth_class_cat)"
    )

    # Build the model using the formula interface so that age, sex,
    # and tooth class are included as covariates.
    model_formula = sm.GLM.from_formula(
        formula,
        data=df_valid,
        family=sm.families.Binomial(),
        freq_weights=df_valid["n_sockets"],
    )
    result = model_formula.fit()

    print(result.summary())

    # Predicted probabilities for each genus at typical covariate values
    ref_age = df_valid["age_at_death"].mean()
    ref_prob_male = df_valid["prob_male_est"].mean()
    ref_tooth = df_valid["tooth_class_cat"].mode()[0]

    genuses = sorted(df_valid["genus_label"].unique())
    pred_rows = []
    for g in genuses:
        pred_rows.append(
            {
                "genus_label": g,
                "age_at_death": ref_age,
                "prob_male_est": ref_prob_male,
                "tooth_class_cat": ref_tooth,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_df["predicted_prop_missing"] = result.predict(pred_df)
    print("Predicted proportion missing by genus at typical covariates:")
    print(pred_df)

    if "Homo sapiens" in pred_df["genus_label"].values:
        human_rate = pred_df.loc[
            pred_df["genus_label"] == "Homo sapiens", "predicted_prop_missing"
        ].iloc[0]
        nonhuman_rates = pred_df.loc[
            pred_df["genus_label"] != "Homo sapiens", "predicted_prop_missing"
        ]
        mean_nonhuman = nonhuman_rates.mean()
        print("Human predicted rate:", human_rate)
        print("Mean non-human predicted rate:", mean_nonhuman)
        print("Difference (human - non-human):", human_rate - mean_nonhuman)
    else:
        print("Homo sapiens not found among genera; cannot compute human vs non-human contrast.")

    print("Genus coefficients relative to reference genus:")
    for name, coef, pval in zip(
        result.params.index, result.params.values, result.pvalues.values
    ):
        if name.startswith("C(genus_label"):
            print(name, "coef", coef, "p", pval)


if __name__ == "__main__":
    main()
