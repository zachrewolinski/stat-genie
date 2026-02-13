import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Construct key derived variables
    df["stu_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["avg_score"] = df[["feature14", "feature15"]].mean(axis=1)
    return df


def run_ols(df: pd.DataFrame, outcome: str, controls: list[str] | None = None):
    if controls is None:
        controls = []
    x_cols = ["stu_teacher_ratio"] + controls
    X = df[x_cols].astype(float)
    X = sm.add_constant(X)
    y = df[outcome].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()
    return model


def summarize_model(model, label: str) -> dict:
    coef = model.params["stu_teacher_ratio"]
    se = model.bse["stu_teacher_ratio"]
    tval = model.tvalues["stu_teacher_ratio"]
    pval = model.pvalues["stu_teacher_ratio"]
    return {
        "model": label,
        "coef_ratio": float(coef),
        "se_ratio": float(se),
        "t_ratio": float(tval),
        "p_ratio": float(pval),
        "r_squared": float(model.rsquared),
        "nobs": int(model.nobs),
    }


def main():
    df = load_data("caschools.csv")

    # Basic correlation between ratio and scores
    corr_read = df["stu_teacher_ratio"].corr(df["feature14"])
    corr_math = df["stu_teacher_ratio"].corr(df["feature15"])
    corr_avg = df["stu_teacher_ratio"].corr(df["avg_score"])

    # Regression models: bivariate and with key controls
    controls = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    models = {
        "read_biv": run_ols(df, "feature14"),
        "math_biv": run_ols(df, "feature15"),
        "avg_biv": run_ols(df, "avg_score"),
        "read_ctrl": run_ols(df, "feature14", controls),
        "math_ctrl": run_ols(df, "feature15", controls),
        "avg_ctrl": run_ols(df, "avg_score", controls),
    }

    summaries = [summarize_model(m, name) for name, m in models.items()]

    results = {
        "n": int(len(df)),
        "corr": {
            "read": float(corr_read),
            "math": float(corr_math),
            "avg": float(corr_avg),
        },
        "models": summaries,
        "ratio_summary": {
            "mean_ratio": float(df["stu_teacher_ratio"].mean()),
            "sd_ratio": float(df["stu_teacher_ratio"].std()),
        },
        "score_summary": {
            "mean_avg_score": float(df["avg_score"].mean()),
            "sd_avg_score": float(df["avg_score"].std()),
        },
    }

    # Save detailed numerical results to a sidecar JSON for inspection.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))

    # Also print a human-readable snapshot to stdout for quick review.
    print("Correlation (ratio vs scores):", results["corr"])
    for s in summaries:
        print(
            f"{s['model']}: coef={s['coef_ratio']:.3f}, "
            f"se={s['se_ratio']:.3f}, t={s['t_ratio']:.2f}, "
            f"p={s['p_ratio']:.4f}, R2={s['r_squared']:.3f}, n={s['nobs']}"
        )


if __name__ == "__main__":
    main()

