import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Basic info
print("rows", len(df))
print("cols", df.columns.tolist())

# Check feature20 vs computed speed variants
# Reading speed computed as words per minute using time on page (feature4) or time minus scrolling (feature5)
speed_calc_f5 = df["feature7"] * 60000 / df["feature5"]
speed_calc_f4 = df["feature7"] * 60000 / df["feature4"]
ms_per_word_f5 = df["feature5"] / df["feature7"]
ms_per_word_f4 = df["feature4"] / df["feature7"]

# Correlations between feature20 and derived metrics
print("corr_feature20_speed_f5", speed_calc_f5.corr(df["feature20"], method="pearson"))
print("corr_feature20_speed_f4", speed_calc_f4.corr(df["feature20"], method="pearson"))
print("corr_feature20_ms_per_word_f5", ms_per_word_f5.corr(df["feature20"], method="pearson"))
print("corr_feature20_ms_per_word_f4", ms_per_word_f4.corr(df["feature20"], method="pearson"))
print("corr_feature20_time_f5", df["feature5"].corr(df["feature20"], method="pearson"))
print("corr_feature20_time_f4", df["feature4"].corr(df["feature20"], method="pearson"))
print("corr_feature20_words", df["feature7"].corr(df["feature20"], method="pearson"))

# Dyslexia subset using feature17 == 1
# Also check distribution of feature12
print("feature17 value counts:\n", df["feature17"].value_counts(dropna=False))
print("feature12 value counts:\n", df["feature12"].value_counts(dropna=False))

sub = df[df["feature17"] == 1].copy()
print("dyslexia rows", len(sub))

# Reading speed proxy: words per minute using time minus scrolling (feature5)
sub = sub[(sub["feature5"] > 0) & (sub["feature4"] > 0)].copy()
sub["wpm_f5"] = sub["feature7"] * 60000 / sub["feature5"]
sub["wpm_f4"] = sub["feature7"] * 60000 / sub["feature4"]

def compare_groups(speed_series, group_series, label):
    rv = speed_series[group_series == 1].astype(float)
    no_rv = speed_series[group_series == 0].astype(float)
    print(f"{label} rv n", rv.shape[0], "no_rv n", no_rv.shape[0])
    print(f"{label} rv mean", rv.mean(), "no_rv mean", no_rv.mean())
    print(f"{label} rv median", rv.median(), "no_rv median", no_rv.median())

    wt = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy='omit')
    print(f"{label} welch_t", wt)

    try:
        mw = stats.mannwhitneyu(rv, no_rv, alternative='two-sided')
        print(f"{label} mannwhitney", mw)
    except Exception as e:
        print(f"{label} mannwhitney error", e)

    n1, n2 = rv.shape[0], no_rv.shape[0]
    mean1, mean2 = rv.mean(), no_rv.mean()
    var1, var2 = rv.var(ddof=1), no_rv.var(ddof=1)
    pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if n1+n2>2 else np.nan
    cohen_d = (mean1 - mean2) / pooled_sd if pooled_sd and pooled_sd>0 else np.nan
    print(f"{label} cohen_d", cohen_d)

    rv_log = np.log1p(rv)
    no_log = np.log1p(no_rv)
    wt_log = stats.ttest_ind(rv_log, no_log, equal_var=False, nan_policy='omit')
    print(f"{label} welch_t_log", wt_log)

    print(f"{label} rv quantiles", rv.quantile([0.1,0.25,0.5,0.75,0.9]).to_dict())
    print(f"{label} no_rv quantiles", no_rv.quantile([0.1,0.25,0.5,0.75,0.9]).to_dict())

    mean_diff = mean1 - mean2
    rel_improvement = mean_diff / mean2 if mean2 != 0 else np.nan
    print(f"{label} mean_diff", mean_diff, "rel_improvement", rel_improvement)

    from scipy.stats import trim_mean
    trim_rv = trim_mean(rv, 0.1)
    trim_no = trim_mean(no_rv, 0.1)
    print(f"{label} trimmed mean rv", trim_rv, "no", trim_no, "diff", trim_rv - trim_no)

compare_groups(sub["wpm_f5"], sub["feature3"], "wpm_f5")
compare_groups(sub["wpm_f4"], sub["feature3"], "wpm_f4")

# Provide counts by reader view for dyslexia
print("dyslexia counts by reader view:\n", sub["feature3"].value_counts())
