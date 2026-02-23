import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen",
            "feature3": "num_missing",
            "feature4": "num_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    df = df[
        (df["num_sockets"] > 0)
        & (df["num_missing"] >= 0)
        & (df["num_missing"] <= df["num_sockets"])
    ].copy()

    return df


def fit_binomial_model(df: pd.DataFrame):
    y = np.column_stack(
        [df["num_missing"].to_numpy(), (df["num_sockets"] - df["num_missing"]).to_numpy()]
    )

    X = pd.get_dummies(
        df[["genus", "tooth_class", "age", "sex"]],
        columns=["genus", "tooth_class"],
        drop_first=True,
    )
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})
    return result, X


def genus_predictions(result, X: pd.DataFrame, df: pd.DataFrame):
    genus_dummy_cols = [c for c in X.columns if c.startswith("genus_")]

    predictions = {}
    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]

    for g in genera:
        X_g = X.copy()
        for col in genus_dummy_cols:
            X_g[col] = 0
        if g != "Homo sapiens":
            col_name = f"genus_{g}"
            if col_name in X_g.columns:
                X_g[col_name] = 1

        probs = result.predict(X_g)
        weights = df["num_sockets"].to_numpy()
        avg_prob = float(np.average(probs, weights=weights))
        predictions[g] = avg_prob

    return predictions


def main():
    csv_path = "amtl.csv"
    df = load_data(csv_path)

    result, X = fit_binomial_model(df)
    print(result.summary())

    preds = genus_predictions(result, X, df)
    print("\nAverage predicted AMTL probability by genus (per socket):")
    for g, p in preds.items():
        print(f"{g}: {p:.4f}")

    genus_coefs = {
        name: (coef, pval)
        for name, coef, pval in zip(
            result.params.index, result.params.to_numpy(), result.pvalues.to_numpy()
        )
        if name.startswith("genus_")
    }
    print("\nGenus coefficients (vs Homo sapiens baseline):")
    for name, (coef, pval) in genus_coefs.items():
        print(f"{name}: coef={coef:.4f}, p-value={pval:.4g}")


if __name__ == "__main__":
    main()

