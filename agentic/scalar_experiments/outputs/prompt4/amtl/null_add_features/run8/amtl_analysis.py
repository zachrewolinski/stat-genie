import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=cols)

    df = df[df["sockets"] > 0]
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])]

    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    df["tooth_class"] = df["tooth_class"].astype("category")
    df["genus"] = df["genus"].astype("category")

    return df


def fit_model(df: pd.DataFrame):
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(result) -> None:
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    print("=== Genus effects relative to Homo sapiens (log-odds of AMTL) ===")
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term not in params:
            print(f"Term {term} not found in model parameters.")
            continue
        coef = params[term]
        ci_low, ci_high = conf_int.loc[term]
        pval = pvalues[term]
        odds_ratio = float(np.exp(coef))
        print(
            f"{genus:5s}: coef={coef: .3f}, OR={odds_ratio: .3f}, "
            f"95% CI=({ci_low: .3f}, {ci_high: .3f}), p={pval: .3g}"
        )


def summarize_predicted_rates(df: pd.DataFrame, result) -> None:
    print("\n=== Predicted AMTL proportions by genus (adjusted) ===")
    # Create a reference covariate profile (median age, mean prob_male, most common tooth_class)
    ref_age = float(df["age"].median())
    ref_prob_male = float(df["prob_male"].mean())
    common_tooth_class = df["tooth_class"].mode().iat[0]

    print(
        f"Using reference profile: age={ref_age:.1f}, "
        f"prob_male={ref_prob_male:.2f}, tooth_class={common_tooth_class}"
    )

    ref_rows = []
    for genus in ["Homo sapiens", "Pan", "Papio", "Pongo"]:
        ref_rows.append(
            {
                "genus": genus,
                "age": ref_age,
                "prob_male": ref_prob_male,
                "tooth_class": common_tooth_class,
            }
        )
    ref_df = pd.DataFrame(ref_rows)

    preds = result.get_prediction(ref_df).summary_frame()
    for genus, mean_pred in zip(ref_df["genus"], preds["mean"]):
        print(f"{genus:12s}: predicted AMTL proportion ≈ {mean_pred:.3f}")


def main() -> None:
    base = Path(".")
    info_path = base / "info.json"
    data_path = base / "amtl.csv"

    metadata = load_metadata(info_path)
    question = metadata.get("research_questions", ["<unknown>"])[0]
    print("Research question:")
    print(question)
    print()

    df = load_data(data_path)
    print(f"Loaded data with {len(df)} rows after cleaning.")
    print("Genus counts:")
    print(df["genus"].value_counts())
    print()

    result = fit_model(df)
    print(result.summary())

    summarize_genus_effects(result)
    summarize_predicted_rates(df, result)


if __name__ == "__main__":
    main()

