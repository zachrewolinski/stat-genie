import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only columns relevant to the research question.
    cols = [
        "win",
        "n_focal",
        "n_other",
        "dist_focal",
        "dist_other",
        "dyad",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df = df[cols].copy()
    df = df.dropna()
    return df


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Relative group size: focal minus other and proportion focal.
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["size_ratio"] = df["n_focal"] / (df["n_focal"] + df["n_other"])

    # Contest location: indicator and distance difference.
    df["focal_closer_home"] = (df["dist_focal"] < df["dist_other"]).astype(int)
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]
    return df


def fit_logit(df: pd.DataFrame):
    # Use a parsimonious model focused on the research question.
    # Predictors: standardized size_ratio and dist_diff plus indicator focal_closer_home.
    model_df = df.copy()
    for col in ["size_ratio", "dist_diff"]:
        mean = model_df[col].mean()
        std = model_df[col].std()
        if std == 0 or np.isnan(std):
            model_df[f"z_{col}"] = model_df[col] - mean
        else:
            model_df[f"z_{col}"] = (model_df[col] - mean) / std

    exog = model_df[["z_size_ratio", "z_dist_diff", "focal_closer_home"]]
    exog = sm.add_constant(exog, has_constant="add")
    endog = model_df["win"]

    logit_model = sm.Logit(endog, exog)
    # Standard maximum-likelihood logistic regression.
    result = logit_model.fit(disp=False)
    return result


def summarize_results(res) -> dict:
    params = res.params
    pvalues = res.pvalues

    summary = {
        "params": params.to_dict(),
        "pvalues": pvalues.to_dict(),
        "n_obs": int(res.nobs),
        "aic": float(res.aic),
    }
    return summary


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "crofoot.csv"
    df = load_data(csv_path)
    df = add_derived_variables(df)
    # Primary model including both relative group size and contest location.
    res_full = fit_logit(df)
    summary_full = summarize_results(res_full)

    # Simpler models to examine each predictor separately.
    # Model 1: win ~ size_ratio
    exog_size = sm.add_constant(df[["size_ratio"]], has_constant="add")
    res_size = sm.Logit(df["win"], exog_size).fit(disp=False)
    summary_size = summarize_results(res_size)

    # Model 2: win ~ dist_diff
    exog_dist = sm.add_constant(df[["dist_diff"]], has_constant="add")
    res_dist = sm.Logit(df["win"], exog_dist).fit(disp=False)
    summary_dist = summarize_results(res_dist)

    out = {
        "full_model": summary_full,
        "size_only": summary_size,
        "dist_only": summary_dist,
    }

    # Save numeric summaries to help manual inspection.
    out_path = base_dir / "analysis_summary.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    # Also print concise textual summaries to stdout.
    print("Full logit model: win ~ size_ratio + dist_diff + focal_closer_home")
    print(f"  N={summary_full['n_obs']}, AIC={summary_full['aic']:.3f}")
    for name in summary_full["params"]:
        coef = summary_full["params"][name]
        pval = summary_full["pvalues"][name]
        print(f"    {name:18s} coef={coef: .3f}, p={pval: .4f}")

    print("\nSize-only model: win ~ size_ratio")
    print(f"  N={summary_size['n_obs']}, AIC={summary_size['aic']:.3f}")
    for name in summary_size["params"]:
        coef = summary_size["params"][name]
        pval = summary_size["pvalues"][name]
        print(f"    {name:18s} coef={coef: .3f}, p={pval: .4f}")

    print("\nDistance-only model: win ~ dist_diff")
    print(f"  N={summary_dist['n_obs']}, AIC={summary_dist['aic']:.3f}")
    for name in summary_dist["params"]:
        coef = summary_dist["params"][name]
        pval = summary_dist["pvalues"][name]
        print(f"    {name:18s} coef={coef: .3f}, p={pval: .4f}")


if __name__ == "__main__":
    main()
