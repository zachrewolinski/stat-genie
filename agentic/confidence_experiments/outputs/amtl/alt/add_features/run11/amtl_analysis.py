import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(".")

    info_path = base_path / "info.json"
    data_path = base_path / "amtl.csv"

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Keep only the genera relevant to the research question.
    genera_of_interest = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Response as proportion of missing teeth, with binomial weights.
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans versus non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Tooth class dummies (reference category is the first alphabetically).
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            df[["is_human", "age", "prob_male"]].reset_index(drop=True),
            tooth_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")

    y = df["amtl_prop"].to_numpy()
    weights = df["sockets"].to_numpy()

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    pval_human = float(result.pvalues["is_human"])

    # Predicted probabilities at mean covariate values, by human vs non-human.
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    # Use the most common tooth class as a representative reference.
    common_class = df["tooth_class"].mode().iloc[0]
    tooth_cols = [c for c in X.columns if c.startswith("tooth_")]
    tooth_vector = {c: 0.0 for c in tooth_cols}
    for c in tooth_cols:
        if c.split("_", 1)[1] == common_class:
            tooth_vector[c] = 1.0

    def build_row(is_human: int) -> np.ndarray:
        row = {
            "const": 1.0,
            "is_human": float(is_human),
            "age": mean_age,
            "prob_male": mean_prob_male,
        }
        row.update(tooth_vector)
        # Ensure columns aligned with model design.
        return np.array([row[col] for col in X.columns], dtype=float)

    row_nonhuman = build_row(0)
    row_human = build_row(1)

    pred_nonhuman = float(result.predict(row_nonhuman[None, :])[0])
    pred_human = float(result.predict(row_human[None, :])[0])
    diff = pred_human - pred_nonhuman

    summary = {
        "research_question": info.get("research_questions", [None])[0],
        "coef_is_human": coef_human,
        "se_is_human": se_human,
        "pval_is_human": pval_human,
        "pred_prob_nonhuman": pred_nonhuman,
        "pred_prob_human": pred_human,
        "pred_prob_diff_human_minus_nonhuman": diff,
        "n_obs": int(df.shape[0]),
        "n_specimens": int(df["specimen"].nunique()),
        "genera_counts": df["genus"].value_counts().to_dict(),
        "common_tooth_class": common_class,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

