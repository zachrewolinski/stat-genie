import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Prefer dyslexia_bin if present; fall back to dyslexia > 0
    if "dyslexia_bin" in df.columns and df["dyslexia_bin"].notna().any():
        dys_df = df[df["dyslexia_bin"] == 1].copy()
    else:
        dys_df = df[df["dyslexia"] > 0].copy()

    # Basic sanity checks
    dys_df = dys_df.dropna(subset=["reader_view", "speed", "uuid", "page_id"])

    # Log-transform speed to reduce skew
    dys_df["log_speed"] = np.log(dys_df["speed"].astype(float))

    # Group summaries
    group_stats = (
        dys_df.groupby("reader_view")["speed"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    # Mixed effects model with random intercept per participant
    model_info = {}
    try:
        model = smf.mixedlm(
            "log_speed ~ reader_view + C(page_id)",
            dys_df,
            groups=dys_df["uuid"],
            re_formula="1",
        )
        fit = model.fit(reml=True, method="lbfgs")
        coef = fit.params.get("reader_view", np.nan)
        pval = fit.pvalues.get("reader_view", np.nan)
        model_info["model"] = "mixedlm"
        model_info["coef"] = coef
        model_info["pval"] = pval
        model_info["converged"] = bool(getattr(fit, "converged", True))
    except Exception as exc:
        # Fallback: OLS with clustered SE by uuid
        model = smf.ols("log_speed ~ reader_view + C(page_id)", data=dys_df)
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": dys_df["uuid"]})
        coef = fit.params.get("reader_view", np.nan)
        pval = fit.pvalues.get("reader_view", np.nan)
        model_info["model"] = "ols_cluster"
        model_info["coef"] = coef
        model_info["pval"] = pval
        model_info["error"] = str(exc)

    # Percent change from log coefficient
    if np.isfinite(model_info["coef"]):
        pct_change = (np.exp(model_info["coef"]) - 1.0) * 100.0
    else:
        pct_change = np.nan

    # Additional context: participant counts by condition
    participants_by_condition = (
        dys_df.groupby("reader_view")["uuid"].nunique().reset_index()
    )

    results = {
        "n_rows": int(dys_df.shape[0]),
        "n_participants": int(dys_df["uuid"].nunique()),
        "group_stats": group_stats.to_dict(orient="records"),
        "participants_by_condition": participants_by_condition.to_dict(orient="records"),
        "model": model_info,
        "pct_change_estimate": pct_change,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
