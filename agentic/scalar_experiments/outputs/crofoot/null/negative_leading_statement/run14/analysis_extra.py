from pathlib import Path
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("crofoot.csv")
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["loc_adv"] = df["dist_focal"] - df["dist_other"]

    for col in ["rel_size", "loc_adv"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0:
            df[f"z_{col}"] = df[col] - mean
        else:
            df[f"z_{col}"] = (df[col] - mean) / std

    y = df["win"]

    # Null model with intercept only.
    X_null = pd.DataFrame({"const": 1.0}, index=df.index)
    X_full = sm.add_constant(df[["z_rel_size", "z_loc_adv"]], has_constant="add")

    model_null = sm.Logit(y, X_null).fit(disp=False)
    model_full = sm.Logit(y, X_full).fit(disp=False)

    lr_stat = 2 * (model_full.llf - model_null.llf)
    df_diff = model_full.df_model - model_null.df_model
    p_value = chi2.sf(lr_stat, df_diff)

    summary = {
        "ll_null": float(model_null.llf),
        "ll_full": float(model_full.llf),
        "lr_stat": float(lr_stat),
        "df_diff": int(df_diff),
        "lr_pvalue": float(p_value),
    }

    Path("lr_test_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
