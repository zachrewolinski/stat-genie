from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] <= df["sockets"]]
    return df


def build_design_matrix(
    df: pd.DataFrame, genus_order: list[str], tooth_order: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df["genus"] = pd.Categorical(df["genus"], categories=genus_order)
    df["tooth_class"] = pd.Categorical(df["tooth_class"], categories=tooth_order)

    genus_dummies = pd.get_dummies(df["genus"], prefix="genus", drop_first=True)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat([df[["age", "prob_male"]], genus_dummies, tooth_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X


def fit_binomial_glm(df: pd.DataFrame):
    # Ensure Homo sapiens is the reference genus
    genera = sorted(df["genus"].unique())
    if "Homo sapiens" not in genera:
        raise ValueError("Homo sapiens not found in genus column.")
    genus_order = ["Homo sapiens"] + [g for g in genera if g != "Homo sapiens"]

    tooth_order = sorted(df["tooth_class"].unique())

    X = build_design_matrix(df, genus_order, tooth_order)
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result, genus_order, tooth_order


def standardized_predictions_by_genus(
    df: pd.DataFrame, result, genus_order: list[str], tooth_order: list[str]
) -> pd.DataFrame:
    """Compute standardized, socket-weighted predicted probabilities for each genus."""
    pred_rows = []

    sockets = df["sockets"].to_numpy()

    for g in genus_order:
        df_g = df.copy()
        df_g["genus"] = g
        X_g = build_design_matrix(df_g, genus_order, tooth_order)
        p = result.predict(X_g)
        # Socket-weighted average probability of a socket being missing
        weighted_mean = float(np.average(p, weights=sockets))
        pred_rows.append({"genus": g, "predicted_prob_missing": weighted_mean})

    return pd.DataFrame(pred_rows)


def summarize_genus_effects(result) -> pd.DataFrame:
    """Extract coefficients, standard errors, z and p for genus terms."""
    params = result.params
    bse = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues

    rows = []
    for name in params.index:
        if name.startswith("genus_"):
            genus = name.split("genus_")[-1]
            coef = params[name]
            z = tvalues[name]
            p = pvalues[name]
            odds_ratio = float(np.exp(coef))
            rows.append(
                {
                    "comparison": f"{genus} vs Homo sapiens",
                    "log_odds_diff": float(coef),
                    "odds_ratio": odds_ratio,
                    "z": float(z),
                    "p_value": float(p),
                }
            )

    return pd.DataFrame(rows)


def main():
    base = Path(__file__).resolve().parent
    csv_path = base / "amtl.csv"

    df = load_data(csv_path)

    print("Unique genera in data:", sorted(df["genus"].unique()))

    result, genus_order, tooth_order = fit_binomial_glm(df)

    print("\n=== GLM summary (truncated) ===")
    print(result.summary().tables[1])

    genus_effects = summarize_genus_effects(result)
    print("\n=== Genus effects (non-human vs Homo sapiens) ===")
    print(genus_effects.to_string(index=False))

    preds = standardized_predictions_by_genus(df, result, genus_order, tooth_order)
    print("\n=== Standardized predicted proportion of missing teeth by genus ===")
    print(preds.to_string(index=False))


if __name__ == "__main__":
    main()
