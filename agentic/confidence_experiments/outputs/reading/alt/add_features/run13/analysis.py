import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Focus on dyslexia participants (binary indicator)
# Use dyslexia_bin == 1; also allow dyslexia >= 1 if dyslexia_bin missing
subset = df.copy()
subset = subset[(subset['dyslexia_bin'] == 1) | ((subset['dyslexia_bin'].isna()) & (subset['dyslexia'] >= 1))]

# Keep only rows with valid speed and reader_view
subset = subset[np.isfinite(subset['speed']) & subset['reader_view'].isin([0, 1])]

# Basic counts
n_total = len(subset)
print(f"Total dyslexia rows: {n_total}")
print("reader_view counts:")
print(subset['reader_view'].value_counts())

# Descriptive stats by reader_view
summary = subset.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
print("\nSpeed by reader_view (raw):")
print(summary)

# Log speed for skew
subset = subset[subset['speed'] > 0].copy()
subset['log_speed'] = np.log(subset['speed'])
log_summary = subset.groupby('reader_view')['log_speed'].agg(['count','mean','median','std'])
print("\nLog speed by reader_view:")
print(log_summary)

# Paired analysis at participant level
# compute per-uuid mean speed by reader_view
pivot = subset.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna()
print(f"\nParticipants with both conditions: {paired.shape[0]}")

if paired.shape[0] > 0:
    diff = paired[1] - paired[0]
    print("Paired mean difference (speed):", diff.mean())
    # paired t-test on log speeds
    pivot_log = subset.pivot_table(index='uuid', columns='reader_view', values='log_speed', aggfunc='mean')
    paired_log = pivot_log.dropna()
    if paired_log.shape[0] > 0:
        tstat, pval = stats.ttest_rel(paired_log[1], paired_log[0])
        print("Paired t-test on log speed: t=%.4f p=%.6f" % (tstat, pval))
        # Wilcoxon signed-rank
        try:
            wstat, wpval = stats.wilcoxon(paired_log[1], paired_log[0])
            print("Wilcoxon signed-rank on log speed: W=%.4f p=%.6f" % (wstat, wpval))
        except Exception as e:
            print("Wilcoxon failed:", e)

# Unpaired test (for completeness)
rv0 = subset.loc[subset['reader_view'] == 0, 'log_speed']
rv1 = subset.loc[subset['reader_view'] == 1, 'log_speed']
if len(rv0) > 1 and len(rv1) > 1:
    tstat2, pval2 = stats.ttest_ind(rv1, rv0, equal_var=False)
    print("\nWelch t-test on log speed (unpaired): t=%.4f p=%.6f" % (tstat2, pval2))
    ustat, upval = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print("Mann-Whitney U on log speed: U=%.4f p=%.6f" % (ustat, upval))

# Mixed effects model with random intercept per participant
# Keep a minimal set of covariates to avoid overfitting
# Use page_id as categorical; num_words and Flesch_Kincaid if present
model_df = subset.dropna(subset=['log_speed','reader_view','uuid','num_words','Flesch_Kincaid','page_id'])
print(f"\nMixedLM rows: {len(model_df)}")

try:
    model = smf.mixedlm(
        "log_speed ~ reader_view + num_words + Flesch_Kincaid + correct_rate + C(page_id)",
        model_df,
        groups=model_df["uuid"]
    )
    result = model.fit(reml=False)
    print(result.summary())
except Exception as e:
    print("MixedLM failed:", e)
