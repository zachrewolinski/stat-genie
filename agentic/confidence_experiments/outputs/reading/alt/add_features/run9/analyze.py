import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main():
    df = pd.read_csv("reading.csv")

    # Identify dyslexic participants (binary indicator preferred)
    if "dyslexia_bin" in df.columns:
        dys = df[df["dyslexia_bin"] == 1].copy()
    else:
        # fallback: dyslexia levels 1 or 2
        dys = df[df["dyslexia"].isin([1, 2])].copy()

    # Drop rows with missing essential values
    dys = dys.dropna(subset=["reader_view", "speed"])

    # Basic group stats
    grp = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).rename(index={0: "no_reader_view", 1: "reader_view"})

    # Two-sample t-test (Welch)
    speed_rv = dys.loc[dys["reader_view"] == 1, "speed"].astype(float)
    speed_no = dys.loc[dys["reader_view"] == 0, "speed"].astype(float)

    t_res = stats.ttest_ind(speed_rv, speed_no, equal_var=False, nan_policy="omit")
    # Mann-Whitney U (non-parametric)
    try:
        u_res = stats.mannwhitneyu(speed_rv, speed_no, alternative="two-sided")
    except ValueError:
        u_res = None

    # Effect size (Cohen's d)
    def cohens_d(a, b):
        a = a.dropna().astype(float)
        b = b.dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        na, nb = len(a), len(b)
        va, vb = a.var(ddof=1), b.var(ddof=1)
        pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
        return (a.mean() - b.mean()) / np.sqrt(pooled)

    d_val = cohens_d(speed_rv, speed_no)

    # Regression with log(speed) to reduce skew, controlling for page length and page_id
    dys = dys.copy()
    dys = dys[dys["speed"] > 0].copy()
    dys["log_speed"] = np.log(dys["speed"])

    # Build design matrix with reader_view + page_id (categorical) + num_words
    X = pd.DataFrame({"reader_view": dys["reader_view"].astype(int)})
    if "num_words" in dys.columns:
        X["num_words"] = dys["num_words"].astype(float)
    if "page_id" in dys.columns:
        page_dummies = pd.get_dummies(dys["page_id"], prefix="page", drop_first=True)
        X = pd.concat([X, page_dummies], axis=1)

    X = sm.add_constant(X, has_constant="add")
    y = dys["log_speed"].astype(float)

    # Cluster-robust SE by participant uuid if available
    if "uuid" in dys.columns:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": dys["uuid"]})
    else:
        model = sm.OLS(y, X).fit()

    # Extract reader_view coefficient
    rv_coef = model.params.get("reader_view", np.nan)
    rv_p = model.pvalues.get("reader_view", np.nan)

    # Convert log-speed coefficient to percent change
    rv_pct = (np.exp(rv_coef) - 1) * 100 if pd.notnull(rv_coef) else np.nan

    results = {
        "n_dyslexic": int(len(dys)),
        "group_stats": grp.reset_index().to_dict(orient="records"),
        "t_test": {"stat": float(t_res.statistic), "p": float(t_res.pvalue)},
        "mannwhitney": None if u_res is None else {"stat": float(u_res.statistic), "p": float(u_res.pvalue)},
        "cohens_d": float(d_val) if pd.notnull(d_val) else None,
        "regression": {
            "reader_view_coef": float(rv_coef) if pd.notnull(rv_coef) else None,
            "reader_view_p": float(rv_p) if pd.notnull(rv_p) else None,
            "reader_view_pct": float(rv_pct) if pd.notnull(rv_pct) else None,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
