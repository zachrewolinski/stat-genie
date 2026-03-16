import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Basic info
print("rows", len(df))
print("columns", df.columns.tolist())

# Identify dyslexia indicator
for col in ["dyslexia_bin", "dyslexia"]:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head())

# Determine speed column
print("speed summary", df["speed"].describe())

# Filter dyslexia individuals
if "dyslexia_bin" in df.columns:
    dys_df = df[df["dyslexia_bin"] == 1].copy()
elif "dyslexia" in df.columns:
    # dyslexia levels 1 or 2
    dys_df = df[df["dyslexia"] > 0].copy()
else:
    raise ValueError("No dyslexia indicator")

print("dyslexia rows", len(dys_df))

# Drop missing speed/reader_view
subset = dys_df.dropna(subset=["speed", "reader_view"])
print("subset rows", len(subset))
print("reader_view counts", subset["reader_view"].value_counts())

# Descriptive stats by reader_view
summary = subset.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()
print("summary by reader_view\n", summary)

# t-test (Welch)
rv0 = subset.loc[subset["reader_view"] == 0, "speed"].values
rv1 = subset.loc[subset["reader_view"] == 1, "speed"].values

if len(rv0) > 1 and len(rv1) > 1:
    t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy="omit")
    print("welch t-test", t_res)

    # nonparametric
    u_res = stats.mannwhitneyu(rv1, rv0, alternative="two-sided")
    print("mannwhitneyu", u_res)

# Mixed effects model with random intercept for uuid, if uuid exists
if "uuid" in subset.columns:
    # log-transform speed for skewness
    subset = subset.copy()
    subset = subset[subset["speed"] > 0]
    subset["log_speed"] = np.log(subset["speed"])

    # Ensure reader_view is treated as categorical (0/1)
    subset["reader_view"] = subset["reader_view"].astype(int)

    # MixedLM: log_speed ~ reader_view + num_words + page_id (optional)
    # Keep simple: reader_view only to avoid convergence issues
    try:
        md = smf.mixedlm("log_speed ~ reader_view", subset, groups=subset["uuid"])
        mdf = md.fit(method="lbfgs", reml=False)
        print(mdf.summary())
    except Exception as e:
        print("MixedLM failed", e)

# Effect size (Cohen's d)
if len(rv0) > 1 and len(rv1) > 1:
    # compute d
    mean0, mean1 = np.mean(rv0), np.mean(rv1)
    s0, s1 = np.var(rv0, ddof=1), np.var(rv1, ddof=1)
    n0, n1 = len(rv0), len(rv1)
    # pooled sd
    sp = np.sqrt(((n0-1)*s0 + (n1-1)*s1) / (n0 + n1 - 2))
    d = (mean1 - mean0) / sp if sp > 0 else np.nan
    print("cohen_d", d)
