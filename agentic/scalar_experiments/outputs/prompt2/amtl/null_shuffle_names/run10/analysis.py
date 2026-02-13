import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Remap column meanings based on metadata in info.json
    # Original CSV columns:
    # sockets, prob_male, genus, age, pop, num_amtl, stdev_age, tooth_class, specimen
    #
    # Semantic mapping from info.json:
    # - sockets: tooth position (Anterior / Posterior / Premolar)
    # - prob_male: specimen identifier
    # - genus (numeric): number of missing teeth of that class
    # - age: number of observable sockets
    # - pop: estimated age at death
    # - num_amtl: uncertainty of age at death
    # - stdev_age: estimated sex of specimen (0–1 scale)
    # - tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
    # - specimen: region
    df = df.copy()
    df["tooth_position"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)
    df["age_at_death"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["sex_estimate"] = df["stdev_age"].astype(float)
    df["genus_label"] = df["tooth_class"]
    df["region"] = df["specimen"]

    # Restrict to the four genera of interest
    genera_of_interest = {"Homo sapiens", "Pan", "Papio", "Pongo"}
    df = df[df["genus_label"].isin(genera_of_interest)].copy()

    # Basic sanity checks
    print("Genus counts:")
    print(df["genus_label"].value_counts(), "\n")

    print("Tooth position counts:")
    print(df["tooth_position"].value_counts(), "\n")

    # Ensure counts are sensible
    valid_mask = (
        df["num_sockets"].notna()
        & df["num_missing"].notna()
        & (df["num_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["num_sockets"])
    )
    invalid_rows = (~valid_mask).sum()
    print(f"Number of rows failing basic count checks: {invalid_rows}")

    df_model = df[valid_mask].copy()

    # Raw AMTL frequencies by genus (pooled across age, sex, tooth position)
    genus_freq = (
        df_model.groupby("genus_label")
        .apply(lambda g: g["num_missing"].sum() / g["num_sockets"].sum())
        .sort_values(ascending=False)
    )
    print("\nRaw AMTL frequency (missing teeth / sockets) by genus:")
    print(genus_freq, "\n")

    # Binary indicator for modern humans vs non-human primates
    df_model["is_human"] = (df_model["genus_label"] == "Homo sapiens").astype(int)

    # Design matrix: intercept + is_human + age at death + sex estimate
    X = pd.DataFrame(index=df_model.index)
    X["intercept"] = 1.0
    X["is_human"] = df_model["is_human"]
    X["age_at_death"] = df_model["age_at_death"]
    X["sex_estimate"] = df_model["sex_estimate"]

    # Add tooth-position indicators (Anterior / Posterior / Premolar, drop one for reference)
    tooth_dummies = pd.get_dummies(
        df_model["tooth_position"], prefix="tooth_pos", drop_first=True
    )
    X = pd.concat([X, tooth_dummies], axis=1)

    # Binomial response as [successes, failures] = [missing, intact]
    endog = np.column_stack(
        [df_model["num_missing"].values, (df_model["num_sockets"] - df_model["num_missing"]).values]
    )

    # Fit binomial GLM
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    print("\nBinomial regression results (AMTL vs sockets):")
    print(result.summary())

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    z = coef / se
    p_value = result.pvalues["is_human"]

    print("\nEffect of being modern human (is_human):")
    print(f"  Coefficient (log-odds): {coef:.4f}")
    print(f"  Std. error: {se:.4f}")
    print(f"  z-statistic: {z:.2f}")
    print(f"  p-value: {p_value:.4g}")

    # Predicted AMTL probability at mean covariate values
    mean_age = df_model["age_at_death"].mean()
    mean_sex = df_model["sex_estimate"].mean()

    base_row = {
        "intercept": 1.0,
        "age_at_death": mean_age,
        "sex_estimate": mean_sex,
    }
    # Tooth position set to the most common category as reference
    for col in tooth_dummies.columns:
        base_row[col] = 0.0

    X_human = base_row.copy()
    X_human["is_human"] = 1.0

    X_nonhuman = base_row.copy()
    X_nonhuman["is_human"] = 0.0

    def predict_prob(row_dict: dict) -> float:
        x_vec = np.array([row_dict[col] for col in X.columns])
        lin_pred = float(np.dot(result.params.values, x_vec))
        return float(1.0 / (1.0 + np.exp(-lin_pred)))

    p_human = predict_prob(X_human)
    p_nonhuman = predict_prob(X_nonhuman)

    print("\nPredicted AMTL probability (at mean age/sex, common tooth position):")
    print(f"  Humans:     {p_human:.4f}")
    print(f"  Non-humans: {p_nonhuman:.4f}")
    print(f"  Difference (human - non-human): {p_human - p_nonhuman:.4f}")


if __name__ == "__main__":
    main()

