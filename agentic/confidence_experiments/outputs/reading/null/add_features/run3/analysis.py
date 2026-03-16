import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "reading.csv"


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Define dyslexic participants
    if "dyslexia_bin" in df.columns:
        df_dys = df[df["dyslexia_bin"] == 1].copy()
    else:
        df_dys = df[df["dyslexia"] > 0].copy()

    # Remove non-positive speeds for log transform
    df_dys = df_dys[df_dys["speed"].notna()].copy()
    df_dys = df_dys[df_dys["speed"] > 0].copy()

    # Group by reader_view
    rv1 = df_dys[df_dys["reader_view"] == 1]["speed"].astype(float)
    rv0 = df_dys[df_dys["reader_view"] == 0]["speed"].astype(float)

    results = {}
    results["n_total_dyslexia"] = int(df_dys.shape[0])
    results["n_reader_view_1"] = int(rv1.shape[0])
    results["n_reader_view_0"] = int(rv0.shape[0])
    results["mean_speed_rv1"] = float(rv1.mean())
    results["mean_speed_rv0"] = float(rv0.mean())
    results["median_speed_rv1"] = float(rv1.median())
    results["median_speed_rv0"] = float(rv0.median())
    results["std_speed_rv1"] = float(rv1.std(ddof=1))
    results["std_speed_rv0"] = float(rv0.std(ddof=1))

    # Welch t-test on raw speed
    t_raw = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    results["welch_t_raw_stat"] = float(t_raw.statistic)
    results["welch_t_raw_p"] = float(t_raw.pvalue)
    results["cohen_d_raw"] = float(cohen_d(rv1, rv0))

    # Welch t-test on log speed
    log_rv1 = np.log(rv1)
    log_rv0 = np.log(rv0)
    t_log = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy="omit")
    results["welch_t_log_stat"] = float(t_log.statistic)
    results["welch_t_log_p"] = float(t_log.pvalue)
    results["cohen_d_log"] = float(cohen_d(log_rv1, log_rv0))
    results["mean_log_rv1"] = float(log_rv1.mean())
    results["mean_log_rv0"] = float(log_rv0.mean())

    # Percent difference in means
    if results["mean_speed_rv0"] != 0:
        results["pct_diff_mean"] = float((results["mean_speed_rv1"] - results["mean_speed_rv0"]) / results["mean_speed_rv0"]) * 100.0
    else:
        results["pct_diff_mean"] = np.nan

    # Simple regression controlling for page_id and num_words (if available)
    # Use log speed for stability
    model_results = {}
    reg_df = df_dys.copy()
    reg_df = reg_df[reg_df["speed"] > 0].copy()
    reg_df["log_speed"] = np.log(reg_df["speed"].astype(float))

    predictors = ["reader_view"]
    if "num_words" in reg_df.columns:
        predictors.append("num_words")
    if "page_id" in reg_df.columns:
        # one-hot encode page_id
        page_dummies = pd.get_dummies(reg_df["page_id"].astype(str), prefix="page", drop_first=True)
        reg_df = pd.concat([reg_df, page_dummies], axis=1)
        predictors.extend(page_dummies.columns.tolist())

    X = reg_df[predictors].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = reg_df["log_speed"].astype(float)
    try:
        ols = sm.OLS(y, X).fit()
        model_results["coef_reader_view"] = float(ols.params.get("reader_view", np.nan))
        model_results["p_reader_view"] = float(ols.pvalues.get("reader_view", np.nan))
        model_results["n_obs"] = int(ols.nobs)
        model_results["r2"] = float(ols.rsquared)
    except Exception as exc:
        model_results["error"] = str(exc)

    results["regression_log_speed"] = model_results

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
