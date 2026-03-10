import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def cohens_d(x, y):
    # pooled SD
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def main():
    df = pd.read_csv("reading.csv")

    # Filter to dyslexia participants (including severe)
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Basic group comparison (independent)
    rv = dys[dys["reader_view"] == 1]["speed"].dropna()
    no = dys[dys["reader_view"] == 0]["speed"].dropna()

    summary = {
        "n_total_dys": int(len(dys)),
        "n_rv": int(len(rv)),
        "n_no": int(len(no)),
        "mean_rv": float(rv.mean()) if len(rv) else np.nan,
        "mean_no": float(no.mean()) if len(no) else np.nan,
        "median_rv": float(rv.median()) if len(rv) else np.nan,
        "median_no": float(no.median()) if len(no) else np.nan,
    }

    # Non-parametric test
    mw_res = None
    if len(rv) > 0 and len(no) > 0:
        mw = stats.mannwhitneyu(rv, no, alternative="two-sided")
        mw_res = {"u": float(mw.statistic), "p": float(mw.pvalue)}

    # t-test on log speed to reduce skew
    t_res = None
    if len(rv) > 1 and len(no) > 1:
        t = stats.ttest_ind(np.log1p(rv), np.log1p(no), equal_var=False)
        t_res = {"t": float(t.statistic), "p": float(t.pvalue)}

    d_raw = cohens_d(rv, no)
    d_log = cohens_d(np.log1p(rv), np.log1p(no)) if len(rv) > 1 and len(no) > 1 else np.nan

    # Within-subject paired comparison if both conditions exist per uuid
    pivot = (
        dys.pivot_table(index="uuid", columns="reader_view", values="speed", aggfunc="mean")
        .rename(columns={0: "no", 1: "rv"})
    )
    paired = pivot.dropna(subset=["no", "rv"])
    paired_res = None
    if len(paired) >= 3:
        # Wilcoxon on paired differences (log1p)
        diff = np.log1p(paired["rv"]) - np.log1p(paired["no"])
        try:
            w = stats.wilcoxon(diff)
            paired_res = {
                "n_paired": int(len(paired)),
                "median_log_diff": float(np.median(diff)),
                "wilcoxon_p": float(w.pvalue),
            }
        except ValueError:
            paired_res = {
                "n_paired": int(len(paired)),
                "median_log_diff": float(np.median(diff)),
                "wilcoxon_p": np.nan,
            }

    # Simple regression controlling for page_id and device (and age) within dyslexia participants
    reg_res = None
    try:
        model_df = dys[["speed", "reader_view", "page_id", "device", "age", "num_words"]].dropna()
        if len(model_df) >= 20:
            # log speed to stabilize
            model_df = model_df.assign(log_speed=np.log1p(model_df["speed"]))
            X = pd.get_dummies(model_df[["reader_view", "page_id", "device", "age", "num_words"]], drop_first=True)
            X = sm.add_constant(X)
            y = model_df["log_speed"]
            model = sm.OLS(y, X).fit()
            coef = model.params.get("reader_view", np.nan)
            pval = model.pvalues.get("reader_view", np.nan)
            reg_res = {"coef_log": float(coef), "p": float(pval), "n": int(len(model_df))}
    except Exception:
        reg_res = None

    results = {
        "summary": summary,
        "mw": mw_res,
        "ttest_log": t_res,
        "cohens_d_raw": float(d_raw) if d_raw is not None else np.nan,
        "cohens_d_log": float(d_log) if d_log is not None else np.nan,
        "paired": paired_res,
        "regression": reg_res,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
