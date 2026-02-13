import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    df = df[df["num_amtl"] <= df["sockets"]]
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])]
    df = df.copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    return df


def fit_binomial_model(df: pd.DataFrame):
    formula = "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens')) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(result, df: pd.DataFrame) -> pd.DataFrame:
    base = {
        "age": df["age"].mean(),
        "prob_male": df["prob_male"].mean(),
        "tooth_class": df["tooth_class"].mode().iloc[0],
    }
    rows = []
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]:
        data = base.copy()
        data["genus"] = genus
        pred = result.get_prediction(pd.DataFrame([data])).summary_frame()
        rows.append(
            {
                "genus": genus,
                "pred_mean": float(pred["mean"].iloc[0]),
                "ci_low": float(pred["mean_ci_lower"].iloc[0]),
                "ci_high": float(pred["mean_ci_upper"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main():
    df = load_and_clean_data("amtl.csv")
    print("Data shape after cleaning:", df.shape)
    print("Genus counts after cleaning:")
    print(df["genus"].value_counts())
    print()

    result = fit_binomial_model(df)
    print(result.summary())

    genus_effects = summarize_genus_effects(result, df)
    print("\nPredicted AMTL proportions by genus (adjusted for age, sex, tooth class):")
    print(genus_effects.sort_values("pred_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

