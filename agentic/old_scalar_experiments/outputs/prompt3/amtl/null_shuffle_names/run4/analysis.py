import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Remap column names to their semantic meaning based on info.json description
    df = df.copy()
    df["tooth_type"] = df["sockets"]  # Anterior / Posterior / Premolar
    df["specimen_id"] = df["prob_male"]  # specimen identifier
    df["num_missing"] = df["genus"].astype(float)  # number of missing teeth of given class
    df["num_sockets"] = df["age"].astype(float)  # observable sockets that could be scored
    df["age_est"] = df["pop"].astype(float)  # estimated age at death
    df["age_sd"] = df["num_amtl"].astype(float)  # uncertainty of age at death
    df["sex_prob_male"] = df["stdev_age"].astype(float)  # probability specimen is male
    df["genus_str"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo
    df["region"] = df["specimen"]

    # Keep only rows for the genera of interest
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus_str"].isin(target_genera)].copy()

    # Drop rows with invalid socket counts
    df = df[df["num_sockets"] > 0].copy()

    # Compute proportion of missing teeth
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["prop_missing", "age_est", "sex_prob_male", "tooth_type", "genus_str"])

    # Crude summary of AMTL by genus (ignoring covariates)
    grouped = df.groupby("genus_str").agg(
        total_missing=("num_missing", "sum"),
        total_sockets=("num_sockets", "sum"),
    )
    grouped["rate_missing"] = grouped["total_missing"] / grouped["total_sockets"]
    print("Crude AMTL rates by genus (missing / sockets):")
    print(grouped.sort_values("rate_missing", ascending=False))
    print()

    # Binomial regression: AMTL ~ genus + age + sex + tooth_type
    # Use proportion response with socket count as frequency weights.
    formula = (
        "prop_missing ~ "
        "C(genus_str, Treatment(reference='Homo sapiens')) + "
        "age_est + sex_prob_male + C(tooth_type)"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    print(result.summary())
    print()

    # Predicted AMTL probabilities for each genus at typical covariate values
    mean_age = df["age_est"].mean()
    mean_sex = df["sex_prob_male"].mean()
    # Use the most common tooth type as reference context
    common_tooth_type = df["tooth_type"].mode().iloc[0]

    pred_rows = []
    for g in target_genera:
        pred_rows.append(
            {
                "genus_str": g,
                "age_est": mean_age,
                "sex_prob_male": mean_sex,
                "tooth_type": common_tooth_type,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_probs = result.predict(pred_df)
    pred_df["pred_amtl_prob"] = pred_probs

    print("Predicted AMTL probabilities at typical covariate values:")
    print(pred_df[["genus_str", "pred_amtl_prob"]].sort_values("pred_amtl_prob", ascending=False))


if __name__ == "__main__":
    main()

