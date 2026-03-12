import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Define dyslexic subset
if 'dyslexia_bin' in df.columns:
    df['dyslexic'] = df['dyslexia_bin'] == 1
else:
    df['dyslexic'] = df['dyslexia'] > 0

# Ensure reader_view is binary 0/1
# Some fields may be floats; coerce

df['reader_view'] = df['reader_view'].astype(float)

# Basic subset for dyslexic

df_dys = df[df['dyslexic']].copy()

# Clean speed values (non-positive or missing)

df_dys = df_dys[np.isfinite(df_dys['speed'])]

df_dys = df_dys[df_dys['speed'] > 0]

# Group stats

groups = df_dys.groupby('reader_view')['speed']
summary = groups.agg(['count','mean','median','std']).rename(index={0.0:'No reader_view',1.0:'Reader_view'})

# Welch t-test
# Ensure both groups present

if set(df_dys['reader_view'].unique()) >= {0.0,1.0}:
    speed0 = df_dys[df_dys['reader_view']==0.0]['speed']
    speed1 = df_dys[df_dys['reader_view']==1.0]['speed']
    t_stat, t_p = stats.ttest_ind(speed1, speed0, equal_var=False, nan_policy='omit')
    # Mann-Whitney U
    try:
        u_stat, u_p = stats.mannwhitneyu(speed1, speed0, alternative='two-sided')
    except ValueError:
        u_stat, u_p = np.nan, np.nan
    # Effect size (Cohen's d)
    n1, n0 = len(speed1), len(speed0)
    sd1, sd0 = speed1.std(ddof=1), speed0.std(ddof=1)
    pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n0-1)*sd0**2) / (n1+n0-2)) if (n1+n0-2) > 0 else np.nan
    cohen_d = (speed1.mean() - speed0.mean()) / pooled_sd if pooled_sd and pooled_sd>0 else np.nan
else:
    t_stat = t_p = u_stat = u_p = cohen_d = np.nan
    speed0 = speed1 = pd.Series(dtype=float)

# Regression: log(speed) ~ reader_view + controls, clustered by uuid
# Controls: page_id, num_words, device, language, age, gender, education, english_native, retake_trial
# Use log1p to reduce skew

reg_results = None

# Build formula with available columns

candidate_controls = ['page_id','num_words','device','language','age','gender','education','english_native','retake_trial']
controls = [c for c in candidate_controls if c in df_dys.columns]

# Drop rows with missing in relevant columns

model_df = df_dys.copy()
model_df = model_df[np.isfinite(model_df['speed']) & (model_df['speed'] > 0)]
model_df['log_speed'] = np.log(model_df['speed'])

# Ensure categories

for cat_col in ['page_id','device','language','education','english_native']:
    if cat_col in model_df.columns:
        model_df[cat_col] = model_df[cat_col].astype('category')

if controls:
    formula = 'log_speed ~ reader_view'
    for c in controls:
        if c in ['page_id','device','language','education','english_native']:
            formula += f' + C({c})'
        else:
            formula += f' + {c}'
else:
    formula = 'log_speed ~ reader_view'

# Fit with cluster-robust SE by uuid if uuid exists

try:
    # Drop rows with missing values in any variable used by the formula
    # This keeps groups aligned with the model data
    # Build list of raw column names used in the formula (excluding categorical wrappers)
    needed_cols = ['log_speed', 'reader_view'] + controls
    model_df = model_df.dropna(subset=[c for c in needed_cols if c in model_df.columns])
    if 'uuid' in model_df.columns:
        reg = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})
    else:
        reg = smf.ols(formula, data=model_df).fit()
    reg_results = reg
except Exception as e:
    reg_results = None
    reg_error = str(e)

# Prepare a small report for manual inspection

report = {
    'n_dyslexic_rows': int(len(df_dys)),
    'reader_view_counts': df_dys['reader_view'].value_counts(dropna=False).to_dict(),
    'summary': summary.to_dict(),
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mann_whitney': {'u_stat': float(u_stat), 'p_value': float(u_p)},
    'cohen_d': float(cohen_d),
}

if reg_results is not None:
    report['regression'] = {
        'formula': formula,
        'coef_reader_view': float(reg_results.params.get('reader_view', np.nan)),
        'se_reader_view': float(reg_results.bse.get('reader_view', np.nan)),
        'p_value_reader_view': float(reg_results.pvalues.get('reader_view', np.nan)),
        'n_obs': int(reg_results.nobs),
        'r2': float(reg_results.rsquared)
    }
else:
    report['regression'] = {'error': reg_error}

print(json.dumps(report, indent=2))
