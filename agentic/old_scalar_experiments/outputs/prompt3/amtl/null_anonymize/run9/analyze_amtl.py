import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic cleaning: drop rows with missing critical fields and require positive sockets
    df = df.dropna(
        subset=["feature1", "feature3", "feature4", "feature5", "feature7", "feature8"]
    )
    df = df[df["feature4"] > 0]
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Response: number of missing teeth out of observable sockets
    successes = df["feature3"].astype(float)
    failures = (df["feature4"] - df["feature3"]).astype(float)
    endog = np.column_stack((successes, failures))

    # Covariates: human vs non-human, age, sex estimate, tooth class
    df = df.copy()
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)
    df["age"] = df["feature5"].astype(float)
    df["sex_est"] = df["feature7"].astype(float)
    df["tooth_class"] = df["feature1"].astype("category")

    X = pd.get_dummies(
        df[["is_human", "age", "sex_est", "tooth_class"]], drop_first=True
    )
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X, df


def summarize_effect(result, X: pd.DataFrame, df: pd.DataFrame) -> dict:
    # Extract coefficient for is_human (effect of humans vs non-humans)
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    z = coef / se
    p_value = result.pvalues["is_human"]

    # Predicted probabilities for humans vs non-humans
    preds = result.predict(X)
    df = df.copy()
    df["pred_prob"] = preds

    human_pred_mean = df.loc[df["is_human"] == 1, "pred_prob"].mean()
    nonhuman_pred_mean = df.loc[df["is_human"] == 0, "pred_prob"].mean()

    human_obs_rate = (
        df.loc[df["is_human"] == 1, "feature3"].sum()
        / df.loc[df["is_human"] == 1, "feature4"].sum()
    )
    nonhuman_obs_rate = (
        df.loc[df["is_human"] == 0, "feature3"].sum()
        / df.loc[df["is_human"] == 0, "feature4"].sum()
    )

    return {
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "z_is_human": float(z),
        "p_is_human": float(p_value),
        "pred_mean_human": float(human_pred_mean),
        "pred_mean_nonhuman": float(nonhuman_pred_mean),
        "obs_rate_human": float(human_obs_rate),
        "obs_rate_nonhuman": float(nonhuman_obs_rate),
    }


def main():
    data_path = Path("amtl.csv")
    df = load_data(data_path)
    result, X, df_model = fit_binomial_model(df)
    summary_stats = summarize_effect(result, X, df_model)

    # Print a compact JSON summary to stdout for inspection
    print(json.dumps(summary_stats, indent=2))


if __name__ == "__main__":
    main()

