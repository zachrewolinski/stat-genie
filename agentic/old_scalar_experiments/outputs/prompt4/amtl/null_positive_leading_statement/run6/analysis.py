import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity filters
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Binary indicator for modern humans vs all non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_binomial_model(df: pd.DataFrame):
    # One row per specimen-tooth_class combination with counts of missing vs present teeth
    y = np.column_stack([df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()])

    # Predictors: human vs non-human, age, sex proxy, tooth class (categorical)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            df[["is_human", "age", "prob_male"]].reset_index(drop=True),
            tooth_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X


def compute_score_from_t(t_value: float) -> int:
    # Map the t-statistic for the human effect to a 0-100 scale using a logistic transform.
    # t = 0 -> 50 (no preference), large positive t -> near 100 (strong "Yes"),
    # large negative t -> near 0 (strong "No").
    score = 100.0 / (1.0 + np.exp(-t_value))
    score_int = int(round(score))
    return max(0, min(100, score_int))


def summarize_effect(df: pd.DataFrame, result, X: pd.DataFrame) -> dict:
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    t_value = coef / se
    p_value = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    # Predict AMTL probability for a "typical" anterior tooth of a human vs non-human,
    # holding continuous covariates at their sample means.
    cov_means = df[["age", "prob_male"]].mean()

    # Identify baseline (anterior) by zeroing all tooth dummies
    tooth_cols = [c for c in X.columns if c.startswith("tooth_")]

    base_row = {"const": 1.0, "is_human": 0.0}
    base_row.update(cov_means.to_dict())
    for col in tooth_cols:
        base_row[col] = 0.0

    non_human_row = pd.Series(base_row)
    human_row = non_human_row.copy()
    human_row["is_human"] = 1.0

    prob_non_human = float(result.predict(non_human_row)[0])
    prob_human = float(result.predict(human_row)[0])

    t_score = float(t_value)
    scalar_score = compute_score_from_t(t_score)

    return {
        "coef": float(coef),
        "se": float(se),
        "t": t_score,
        "p": float(p_value),
        "odds_ratio": odds_ratio,
        "prob_non_human": prob_non_human,
        "prob_human": prob_human,
        "response_scalar": scalar_score,
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result, X = fit_binomial_model(df)
    summary = summarize_effect(df, result, X)

    # Print a concise textual summary for inspection.
    print("Human vs non-human AMTL effect (binomial GLM):")
    print(f"  Coefficient (is_human): {summary['coef']:.3f}")
    print(f"  Std. error:             {summary['se']:.3f}")
    print(f"  t-statistic:            {summary['t']:.3f}")
    print(f"  p-value:                {summary['p']:.3g}")
    print(f"  Odds ratio:             {summary['odds_ratio']:.3f}")
    print(
        "  Predicted AMTL probability (anterior tooth, mean age/sex): "
        f"{summary['prob_non_human']:.3f} (non-human) vs {summary['prob_human']:.3f} (human)"
    )
    print(f"  Derived 0-100 response scalar: {summary['response_scalar']}")

    # Also save the numeric summary to a JSON file for reference.
    with open("analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

