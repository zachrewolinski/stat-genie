import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop rows with missing key fields
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "tooth_class",
            "genus",
        ]
    )

    # Ensure numeric integer counts
    df["num_amtl"] = df["num_amtl"].astype(int)
    df["sockets"] = df["sockets"].astype(int)
    df = df[df["sockets"] > 0].copy()

    # Restrict to genera of interest
    genera_keep = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_keep)].copy()

    # Convert to categorical with explicit ordering
    df["genus"] = pd.Categorical(
        df["genus"], categories=["Papio", "Pan", "Pongo", "Homo sapiens"]
    )
    df["tooth_class"] = pd.Categorical(
        df["tooth_class"], categories=["Anterior", "Premolar", "Posterior"]
    )
    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        n_missing = int(row["num_amtl"])
        n_sockets = int(row["sockets"])
        if n_sockets <= 0:
            continue
        n_missing = max(0, min(n_missing, n_sockets))

        base = row.to_dict()

        for _ in range(n_missing):
            r = dict(base)
            r["amtl"] = 1
            rows.append(r)
        for _ in range(n_sockets - n_missing):
            r = dict(base)
            r["amtl"] = 0
            rows.append(r)

    tooth_df = pd.DataFrame(rows)
    return tooth_df


def fit_model(tooth_df: pd.DataFrame):
    formula = "amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(formula=formula, data=tooth_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def marginal_predicted_probabilities(result, tooth_df: pd.DataFrame) -> pd.Series:
    # Average predicted probability for each genus, averaging over observed
    # distribution of age, sex, and tooth class.
    genera = sorted(tooth_df["genus"].unique())
    preds = {}
    for g in genera:
        tmp = tooth_df.copy()
        tmp["genus"] = g
        preds[str(g)] = result.predict(tmp).mean()
    return pd.Series(preds)


def main():
    df = load_and_prepare_data("amtl.csv")
    tooth_df = expand_to_tooth_level(df)

    result = fit_model(tooth_df)
    print("Model summary:")
    print(result.summary())

    probs = marginal_predicted_probabilities(result, tooth_df)
    print("\nMarginal predicted AMTL probabilities by genus (adjusted):")
    for genus, p in probs.items():
        print(f"{genus}: {p:.4f}")

    human_prob = probs.get("Homo sapiens", np.nan)
    non_human_probs = probs.drop(labels=["Homo sapiens"])
    mean_non_human = non_human_probs.mean()

    print(f"\nHomo sapiens predicted AMTL probability: {human_prob:.4f}")
    print(f"Mean non-human predicted AMTL probability: {mean_non_human:.4f}")
    print("Difference (Homo sapiens - mean non-human): "
          f"{(human_prob - mean_non_human):.4f}")


if __name__ == "__main__":
    main()
