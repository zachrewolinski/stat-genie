import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop rows with any missing values in key fields
    key_cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=key_cols).copy()
    # Remove rows where sockets <= 0 to avoid invalid binomial totals
    df = df[df["sockets"] > 0].copy()
    return df


def fit_binomial_model(df: pd.DataFrame):
    df = df.copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # One-hot encode tooth class (baseline = Anterior)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            df[["is_human", "age", "prob_male"]].reset_index(drop=True),
            tooth_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")

    # Binomial response: successes = num_amtl, failures = sockets - num_amtl
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X, df


def summarize_results(result, X: pd.DataFrame):
    params = result.params
    b_human = params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)

    # Average predicted AMTL probabilities if all were human vs all non-human,
    # holding age, sex, and tooth class at their observed values.
    X_human = X.copy()
    X_human["is_human"] = 1
    X_nonhuman = X.copy()
    X_nonhuman["is_human"] = 0

    p_hat_human = result.predict(X_human).mean()
    p_hat_nonhuman = result.predict(X_nonhuman).mean()
    diff = p_hat_human - p_hat_nonhuman

    summary = {
        "coef_is_human": float(b_human),
        "se_is_human": float(se_human),
        "pvalue_is_human": float(p_human),
        "avg_pred_amtl_human": float(p_hat_human),
        "avg_pred_amtl_nonhuman": float(p_hat_nonhuman),
        "avg_pred_difference": float(diff),
    }
    return summary


def compute_raw_rates(df: pd.DataFrame):
    # Simple observed AMTL rate by genus (num_amtl / sockets)
    grouped = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(rate=lambda g: g["num_amtl"] / g["sockets"])
    )
    return grouped.reset_index()


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    # Raw descriptive rates by genus
    raw_rates = compute_raw_rates(df)

    # Fit binomial regression
    result, X, df_model = fit_binomial_model(df)
    model_summary = summarize_results(result, X)

    # Print key outputs for inspection
    print("=== Raw AMTL rates by genus (num_amtl / sockets) ===")
    print(raw_rates.to_string(index=False))
    print()

    print("=== Binomial logistic regression results ===")
    print(f"coef(is_human) = {model_summary['coef_is_human']:.3f}")
    print(f"se(is_human)   = {model_summary['se_is_human']:.3f}")
    print(f"p-value        = {model_summary['pvalue_is_human']:.3g}")
    print()
    print("=== Adjusted predicted AMTL probabilities ===")
    print(f"Average predicted AMTL (all human)     = {model_summary['avg_pred_amtl_human']:.4f}")
    print(f"Average predicted AMTL (all non-human) = {model_summary['avg_pred_amtl_nonhuman']:.4f}")
    print(f"Difference (human - non-human)         = {model_summary['avg_pred_difference']:.4f}")

    # Also dump a small JSON with the key quantitative results for downstream use if needed
    out = {
        "raw_rates_by_genus": raw_rates.to_dict(orient="records"),
        "model_summary": model_summary,
    }
    Path("analysis_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

