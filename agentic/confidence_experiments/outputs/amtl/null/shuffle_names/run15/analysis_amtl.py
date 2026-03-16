import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Map shuffled column names to their semantic meaning based on info.json:
    # sockets -> tooth_class (Anterior/Posterior/Premolar)
    # prob_male -> specimen id (unused for modeling)
    # genus -> num_amtl (number of missing teeth)
    # age -> sockets (number of observable sockets)
    # pop -> age_at_death
    # num_amtl -> stdev_age (age uncertainty; unused)
    # stdev_age -> prob_male (probability specimen is male)
    # tooth_class -> genus (Homo sapiens, Pan, Papio, Pongo)
    # specimen -> population/region (unused)

    df = df.copy()
    df["tooth_class_sem"] = df["sockets"]
    df["n_missing"] = df["genus"]
    df["n_sockets"] = df["age"]
    df["age_at_death"] = df["pop"]
    df["prob_male_sem"] = df["stdev_age"]
    df["genus_sem"] = df["tooth_class"]

    # Exclude rows with zero observable sockets to avoid invalid proportions
    df = df[df["n_sockets"] > 0].copy()

    # Response for binomial model: successes (missing teeth) and failures (present teeth)
    df["n_present"] = df["n_sockets"] - df["n_missing"]
    df = df[(df["n_missing"] >= 0) & (df["n_present"] >= 0)].copy()

    # Focus on genera relevant to the research question
    mask_genera = df["genus_sem"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])
    df = df[mask_genera].copy()

    # Indicator for human vs non-human primates
    df["is_human"] = (df["genus_sem"] == "Homo sapiens").astype(int)

    # Tooth class categorical (Anterior/Posterior/Premolar)
    df["tooth_class_sem"] = df["tooth_class_sem"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Construct design matrix with covariates: human indicator, age, prob_male, tooth class dummies
    X = pd.DataFrame(
        {
            "intercept": 1.0,
            "is_human": df["is_human"].astype(float),
            "age_at_death": df["age_at_death"].astype(float),
            "prob_male": df["prob_male_sem"].astype(float),
        }
    )
    tooth_dummies = pd.get_dummies(df["tooth_class_sem"], prefix="tooth", drop_first=True)
    X = pd.concat([X, tooth_dummies], axis=1)

    y = np.asarray(list(zip(df["n_missing"].astype(int), df["n_present"].astype(int))))

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_effect(result) -> dict:
    # Extract coefficient and standard error for human indicator
    coef = result.params.get("is_human", np.nan)
    se = result.bse.get("is_human", np.nan)

    # Wald z-statistic and two-sided p-value from the fitted model
    z = result.tvalues.get("is_human", np.nan)
    p_value = result.pvalues.get("is_human", np.nan)

    # Approximate odds ratio
    odds_ratio = float(np.exp(coef))

    summary = {
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "z_is_human": float(z),
        "p_value_is_human": float(p_value),
        "odds_ratio_is_human": odds_ratio,
    }
    return summary


def main():
    base = Path(__file__).parent
    info = load_metadata(base / "info.json")
    df_raw = load_data(base / "amtl.csv")
    df = prepare_data(df_raw)

    result = fit_binomial_model(df)
    effect_summary = summarize_effect(result)

    # Persist a small JSON summary so we can inspect from outside if needed
    out = {
        "n_rows_used": int(df.shape[0]),
        "effect_summary": effect_summary,
    }
    (base / "analysis_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
