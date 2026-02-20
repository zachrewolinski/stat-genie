import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df = df[df["feature4"] > 0].reset_index(drop=True)
    df["prop_missing"] = df["feature3"] / df["feature4"]
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)
    return df


def fit_glm(df: pd.DataFrame):
    """
    Binomial GLM on specimen-level proportions with frequency weights.
    """
    formula = "prop_missing ~ is_human + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()
    return result


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each specimen row into one row per observable socket.
    Each tooth is coded as missing (1) or present (0).
    """
    records = []
    for _, row in df.iterrows():
        n_missing = int(row["feature3"])
        n_sockets = int(row["feature4"])
        n_present = n_sockets - n_missing
        base = {
            "feature1": row["feature1"],
            "feature5": row["feature5"],
            "feature7": row["feature7"],
            "is_human": row["is_human"],
        }
        # Missing teeth
        for _ in range(n_missing):
            rec = base.copy()
            rec["tooth_missing"] = 1
            records.append(rec)
        # Present teeth
        for _ in range(n_present):
            rec = base.copy()
            rec["tooth_missing"] = 0
            records.append(rec)
    tooth_df = pd.DataFrame.from_records(records)
    return tooth_df


def fit_logit_tooth_level(tooth_df: pd.DataFrame):
    formula = "tooth_missing ~ is_human + feature5 + feature7 + C(feature1)"
    model = smf.logit(formula=formula, data=tooth_df)
    result = model.fit(disp=False)
    return result


def summarize_effect(result, var_name: str):
    coef = result.params[var_name]
    conf_int = result.conf_int().loc[var_name].tolist()
    p_value = result.pvalues[var_name]
    odds_ratio = float(np.exp(coef))
    return {
        "coef": float(coef),
        "conf_int": [float(conf_int[0]), float(conf_int[1])],
        "p_value": float(p_value),
        "odds_ratio": odds_ratio,
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} specimen rows")
    print(df[["feature1", "feature3", "feature4", "feature5", "feature7", "feature8"]].head())

    # Specimen-level GLM
    glm_result = fit_glm(df)
    print("\n=== Specimen-level GLM (proportion missing) ===")
    print(glm_result.summary())
    glm_metrics = summarize_effect(glm_result, "is_human")
    print("\nEffect of being human (GLM, specimen level):")
    print(json.dumps(glm_metrics, indent=2))

    # Tooth-level logistic regression
    tooth_df = expand_to_tooth_level(df)
    print(f"\nExpanded to {len(tooth_df)} tooth-level rows")
    logit_result = fit_logit_tooth_level(tooth_df)
    print("\n=== Tooth-level logistic regression ===")
    print(logit_result.summary())
    logit_metrics = summarize_effect(logit_result, "is_human")
    print("\nEffect of being human (logit, tooth level):")
    print(json.dumps(logit_metrics, indent=2))


if __name__ == "__main__":
    main()

