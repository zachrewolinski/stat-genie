import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: ensure positive socket counts
    df = df[df["sockets"] > 0].copy()

    # Drop any logically invalid binomial rows (more missing teeth than sockets, or negative counts)
    invalid_mask = (df["num_amtl"] < 0) | (df["num_amtl"] > df["sockets"])
    if invalid_mask.any():
        print(f"Dropping {invalid_mask.sum()} rows with num_amtl outside [0, sockets].")
        df = df.loc[~invalid_mask].copy()

    # Expand to per-tooth data so we can run a standard logistic regression
    records = []
    for _, row in df.iterrows():
        n_sockets = int(row["sockets"])
        n_missing = int(row["num_amtl"])
        # Clamp for safety after the invalid filter
        n_missing = max(0, min(n_missing, n_sockets))
        n_present = n_sockets - n_missing

        base = {
            "age": row["age"],
            "prob_male": row["prob_male"],
            "genus": row["genus"],
            "tooth_class": row["tooth_class"],
        }

        records.extend({**base, "amtl": 1} for _ in range(n_missing))
        records.extend({**base, "amtl": 0} for _ in range(n_present))

    df_long = pd.DataFrame.from_records(records)

    # Treat categorical variables explicitly and set Homo sapiens as genus reference
    df_long["genus"] = pd.Categorical(
        df_long["genus"],
        categories=["Homo sapiens", "Pan", "Papio", "Pongo"],
    )
    df_long["tooth_class"] = df_long["tooth_class"].astype("category")

    model = smf.logit("amtl ~ genus + age + prob_male + tooth_class", data=df_long)
    result = model.fit(disp=False)

    print(result.summary())

    # Crude (unadjusted) AMTL rates by genus based on the original aggregated data
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(rate=lambda x: x["num_amtl"] / x["sockets"])
    )
    print("\nUnadjusted AMTL rates by genus (num_amtl / sockets):")
    print(genus_summary)

    # Adjusted predicted AMTL probabilities by genus:
    genus_levels = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    avg_preds = {}
    for g in genus_levels:
        new_data = df_long.copy()
        new_data["genus"] = g
        preds = result.predict(new_data)
        avg_preds[g] = preds.mean()

    print("\nAdjusted mean AMTL probabilities by genus (holding age/sex/tooth_class distribution constant):")
    for g, p in avg_preds.items():
        print(f"{g}: {p:.3f}")

    print("\nGenus coefficients relative to Homo sapiens:")
    for param_name, coef in result.params.items():
        if param_name.startswith("genus[T."):
            p_value = result.pvalues[param_name]
            print(f"{param_name}: coef={coef:.3f}, p={p_value:.4g}")


if __name__ == "__main__":
    main()
