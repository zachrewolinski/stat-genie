import json
import math
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def load_data(path):
    df = pd.read_csv(path)
    return df


def pick_dyslexia_subset(df):
    if "dyslexia_bin" in df.columns:
        subset = df[df["dyslexia_bin"] == 1].copy()
        subset["dyslexia_flag"] = 1
    else:
        subset = df[df["dyslexia"] > 0].copy()
        subset["dyslexia_flag"] = 1
    return subset


def summarize_by_reader_view(df):
    summary = (
        df.groupby("reader_view")["speed"]
        .agg(["count", "mean", "median", "std"])
        .rename_axis("reader_view")
        .reset_index()
    )
    return summary


def welch_ttest_log_speed(df):
    df = df.copy()
    df = df[df["speed"] > 0]
    df["log_speed"] = np.log(df["speed"])
    rv1 = df[df["reader_view"] == 1]["log_speed"]
    rv0 = df[df["reader_view"] == 0]["log_speed"]
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    # effect in log units and geometric mean ratio
    diff = rv1.mean() - rv0.mean()
    ratio = math.exp(diff)
    return {
        "tstat": tstat,
        "pval": pval,
        "log_diff": diff,
        "geo_ratio": ratio,
    }


def mann_whitney_speed(df):
    rv1 = df[df["reader_view"] == 1]["speed"].dropna()
    rv0 = df[df["reader_view"] == 0]["speed"].dropna()
    # Use two-sided test
    ustat, pval = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    return {"ustat": ustat, "pval": pval}


def cluster_ols_log_speed(df):
    df = df.copy()
    df = df[df["speed"] > 0]
    df["log_speed"] = np.log(df["speed"])
    # Control for page_id and num_words, Flesch_Kincaid to account for text difficulty.
    # Cluster robust SE by uuid.
    if "Flesch_Kincaid" in df.columns:
        formula = "log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid"
    else:
        formula = "log_speed ~ reader_view + C(page_id) + num_words"
    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["uuid"]})
    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)
    return {
        "coef": coef,
        "pval": pval,
        "model": model,
    }


def mixedlm_log_speed(df):
    df = df.copy()
    df = df[df["speed"] > 0]
    df["log_speed"] = np.log(df["speed"])
    # Mixed effects: random intercept by uuid, variance component for page_id.
    if "Flesch_Kincaid" in df.columns:
        formula = "log_speed ~ reader_view + num_words + Flesch_Kincaid"
    else:
        formula = "log_speed ~ reader_view + num_words"
    vc = {"page": "0 + C(page_id)"}
    try:
        model = smf.mixedlm(formula, data=df, groups=df["uuid"], vc_formula=vc)
        result = model.fit(reml=False, method="lbfgs", maxiter=200)
    except Exception as exc:
        return {"error": str(exc)}
    coef = result.params.get("reader_view", np.nan)
    pval = result.pvalues.get("reader_view", np.nan)
    return {
        "coef": coef,
        "pval": pval,
        "result": result,
    }


def main():
    df = load_data("reading.csv")
    dys_df = pick_dyslexia_subset(df)

    # Basic counts
    counts = {
        "rows_total": len(df),
        "rows_dyslexia": len(dys_df),
        "unique_uuid_dyslexia": dys_df["uuid"].nunique(),
    }

    summary = summarize_by_reader_view(dys_df)
    ttest = welch_ttest_log_speed(dys_df)
    mwu = mann_whitney_speed(dys_df)
    ols = cluster_ols_log_speed(dys_df)
    mixed = mixedlm_log_speed(dys_df)

    # Export key results to json for later use
    out = {
        "counts": counts,
        "summary": summary.to_dict(orient="records"),
        "welch_ttest_log_speed": ttest,
        "mann_whitney_speed": mwu,
        "cluster_ols_log_speed": {"coef": ols["coef"], "pval": ols["pval"]},
        "mixedlm_log_speed": {"coef": mixed.get("coef"), "pval": mixed.get("pval"), "error": mixed.get("error")},
    }

    with open("analysis_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print concise summary to stdout
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
