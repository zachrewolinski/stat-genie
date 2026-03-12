import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def cohens_dz(diffs):
    diffs = np.asarray(diffs)
    return diffs.mean() / diffs.std(ddof=1)


def main():
    df = pd.read_csv("reading.csv")

    # Focus on participants with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()
    dys = dys[dys["speed"] > 0].copy()

    dys["log_speed"] = np.log(dys["speed"])

    n_rows = len(dys)
    n_uuid = dys["uuid"].nunique()

    group_stats = (
        dys.groupby("reader_view")["speed"]
        .agg(["count", "mean", "median", "std"])
        .rename_axis("reader_view")
    )

    rv1 = dys.loc[dys["reader_view"] == 1, "log_speed"]
    rv0 = dys.loc[dys["reader_view"] == 0, "log_speed"]

    # Independent groups tests
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")

    d_log = cohens_d(rv1, rv0)

    gm1 = np.exp(rv1.mean())
    gm0 = np.exp(rv0.mean())
    gm_ratio = gm1 / gm0

    # Paired analysis at participant level (mean per condition)
    per = (
        dys.groupby(["uuid", "reader_view"])["log_speed"]
        .mean()
        .reset_index()
    )
    pivot = per.pivot(index="uuid", columns="reader_view", values="log_speed")
    paired = pivot.dropna()
    paired_t = None
    paired_p = None
    paired_dz = None
    paired_ratio = None
    if not paired.empty:
        diffs = paired[1] - paired[0]
        paired_t, paired_p = stats.ttest_rel(paired[1], paired[0])
        paired_dz = cohens_dz(diffs)
        paired_ratio = np.exp(diffs.mean())

    # Regression with cluster-robust SE by participant
    # Keep rows with required covariates
    covars = [
        "reader_view",
        "page_id",
        "num_words",
        "Flesch_Kincaid",
        "device",
        "age",
        "gender",
        "education",
        "english_native",
        "uuid",
        "log_speed",
    ]
    reg = dys[covars].dropna().copy()
    reg_model = smf.ols(
        "log_speed ~ reader_view + C(page_id) + num_words + Flesch_Kincaid + C(device) + age + C(gender) + C(education) + C(english_native)",
        data=reg,
    ).fit(cov_type="cluster", cov_kwds={"groups": reg["uuid"]})

    rv_coef = reg_model.params.get("reader_view", np.nan)
    rv_p = reg_model.pvalues.get("reader_view", np.nan)

    results = {
        "n_rows": int(n_rows),
        "n_uuid": int(n_uuid),
        "group_stats": group_stats.reset_index().to_dict(orient="records"),
        "welch_t": {"t": float(t_stat), "p": float(t_p)},
        "mann_whitney": {"u": float(u_stat), "p": float(u_p)},
        "cohens_d_log": float(d_log),
        "geom_mean_speed": {"rv1": float(gm1), "rv0": float(gm0), "ratio": float(gm_ratio)},
        "paired": {
            "n_uuid": int(paired.shape[0]),
            "t": None if paired_t is None else float(paired_t),
            "p": None if paired_p is None else float(paired_p),
            "dz": None if paired_dz is None else float(paired_dz),
            "ratio": None if paired_ratio is None else float(paired_ratio),
        },
        "regression": {
            "n_rows": int(reg.shape[0]),
            "coef_reader_view": float(rv_coef),
            "p_reader_view": float(rv_p),
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
