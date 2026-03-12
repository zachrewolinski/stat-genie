import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Define dyslexia subset: dyslexia_bin==1 OR dyslexia>=1
# Preserve missing values; treat >0.5 as dyslexia for bin field
_df['dyslexia_bin'] = pd.to_numeric(_df['dyslexia_bin'], errors='coerce')

# Create log speed to handle skew; add small constant
_df['log_speed'] = np.log(_df['speed'].clip(lower=1e-6))

# Subsets
subset_bin = _df[_df['dyslexia_bin'] == 1].copy()
subset_ge1 = _df[_df['dyslexia'] >= 1].copy()

results = {}

def summarize(sub, label):
    out = {}
    out['n_rows'] = len(sub)
    out['n_participants'] = sub['uuid'].nunique()
    out['mean_speed_reader_view_1'] = sub.loc[sub['reader_view'] == 1, 'speed'].mean()
    out['mean_speed_reader_view_0'] = sub.loc[sub['reader_view'] == 0, 'speed'].mean()
    out['median_speed_reader_view_1'] = sub.loc[sub['reader_view'] == 1, 'speed'].median()
    out['median_speed_reader_view_0'] = sub.loc[sub['reader_view'] == 0, 'speed'].median()
    # OLS with page_id fixed effects, cluster by participant
    # Model: log_speed ~ reader_view + C(page_id) + C(language) + C(device)
    # Include language/device to control for obvious confounds.
    try:
        model_cols = ['log_speed', 'reader_view', 'page_id', 'language', 'device', 'uuid']
        sub_model = sub[model_cols].dropna()
        model = smf.ols('log_speed ~ reader_view + C(page_id) + C(language) + C(device)', data=sub_model)
        fit = model.fit(cov_type='cluster', cov_kwds={'groups': sub_model['uuid']})
        out['ols_coef_reader_view'] = float(fit.params['reader_view'])
        out['ols_p_reader_view'] = float(fit.pvalues['reader_view'])
        out['ols_ci_low'] = float(fit.conf_int().loc['reader_view', 0])
        out['ols_ci_high'] = float(fit.conf_int().loc['reader_view', 1])
        out['ols_n'] = int(fit.nobs)
    except Exception as e:
        out['ols_error'] = str(e)
    return out

results['dyslexia_bin_1'] = summarize(subset_bin, 'dyslexia_bin_1')
results['dyslexia_ge1'] = summarize(subset_ge1, 'dyslexia_ge1')

# Also compute paired within-participant difference if both conditions exist
pair_stats = {}
for name, sub in [('dyslexia_bin_1', subset_bin), ('dyslexia_ge1', subset_ge1)]:
    # compute per-participant mean log_speed by reader_view
    pivot = sub.pivot_table(index='uuid', columns='reader_view', values='log_speed', aggfunc='mean')
    pivot = pivot.dropna()  # keep participants with both conditions
    pair_stats[name] = {
        'n_participants_both': int(len(pivot)),
        'mean_log_speed_diff_rv1_minus_rv0': float((pivot[1] - pivot[0]).mean()) if len(pivot) else np.nan,
        'median_log_speed_diff_rv1_minus_rv0': float((pivot[1] - pivot[0]).median()) if len(pivot) else np.nan,
    }

results['pair_stats'] = pair_stats

# Save results
pd.Series(results).to_json('analysis_results.json', indent=2)

print('Saved analysis_results.json')
