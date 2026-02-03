import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Focus on participants with dyslexia (binary indicator).
# Handle cases where columns are not cleanly binary by using a reasonable threshold.
def _is_binary(series):
    vals = pd.Series(series.dropna().unique())
    return set(vals).issubset({0, 1})

if 'dyslexia_bin' in _df.columns and _is_binary(_df['dyslexia_bin']):
    dys_flag = _df['dyslexia_bin'] == 1
elif 'dyslexia' in _df.columns and set(pd.Series(_df['dyslexia'].dropna().unique())).issubset({0, 1, 2}):
    dys_flag = _df['dyslexia'] >= 1
elif 'dyslexia' in _df.columns:
    # Fallback threshold for noisy continuous data
    dys_flag = _df['dyslexia'] >= 1.0
else:
    dys_flag = _df['dyslexia_bin'] >= 0.5

dys_df = _df[dys_flag].copy()

# Basic sanity
print(f"Total rows: {len(_df)}, Dyslexia rows: {len(dys_df)}")

# Drop rows with missing key fields
key_cols = ['reader_view', 'speed']
dys_df = dys_df.dropna(subset=key_cols)

# Ensure reader_view is binary
if not _is_binary(dys_df['reader_view']):
    dys_df['reader_view'] = (dys_df['reader_view'] >= 0.5).astype(int)

# Summary stats
summary = dys_df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])
print('\nSpeed summary for dyslexia group by reader_view (0=off,1=on):')
print(summary)

# Compute mean difference
mean_diff = summary.loc[1, 'mean'] - summary.loc[0, 'mean']
print(f"\nMean speed difference (reader_view=1 minus 0): {mean_diff:.3f}")

# t-test (Welch)
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']

ttest_res = sm.stats.ttest_ind(rv1, rv0, usevar='unequal')
print(f"\nWelch t-test: t={ttest_res[0]:.3f}, p={ttest_res[1]:.4g}, df={ttest_res[2]:.1f}")

# Log-speed regression with controls to reduce skew/outliers
# Add small constant to avoid log(0)
log_df = dys_df.copy()
log_df['log_speed'] = np.log(log_df['speed'] + 1e-6)

# Build a modest control model
# Use available controls if columns exist
controls = []
for col in ['num_words', 'age', 'Flesch_Kincaid']:
    if col in log_df.columns:
        controls.append(col)

# Categorical controls
cat_controls = []
for col in ['page_id', 'device', 'education', 'language', 'english_native']:
    if col in log_df.columns:
        cat_controls.append(f"C({col})")

rhs = ['reader_view'] + controls + cat_controls
formula = 'log_speed ~ ' + ' + '.join(rhs)

model = smf.ols(formula, data=log_df).fit()
print('\nLog-speed regression (dyslexia group):')
print(model.summary().tables[1])

# Also compute percent change from reader_view coefficient
coef = model.params.get('reader_view', np.nan)
if np.isfinite(coef):
    pct = (np.exp(coef) - 1) * 100
    print(f"\nReader_view coefficient on log_speed: {coef:.4f} (~{pct:.2f}% change)")

# Save key results for later use in conclusion
results = {
    'n_dys': len(dys_df),
    'mean_speed_rv1': summary.loc[1, 'mean'],
    'mean_speed_rv0': summary.loc[0, 'mean'],
    'mean_diff': mean_diff,
    'ttest_p': ttest_res[1],
    'reg_coef': coef,
    'reg_p': model.pvalues.get('reader_view', np.nan),
}

pd.Series(results).to_csv('analysis_results.csv')
