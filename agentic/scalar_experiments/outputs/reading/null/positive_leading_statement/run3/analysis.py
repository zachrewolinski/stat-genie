import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = 'reading.csv'

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure numeric where needed
for col in ['reader_view', 'speed', 'dyslexia', 'dyslexia_bin', 'num_words', 'age']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter to dyslexic participants (binary flag)
if 'dyslexia_bin' in df.columns:
    df_dys = df[df['dyslexia_bin'] == 1].copy()
else:
    df_dys = df[df['dyslexia'] >= 1].copy()

# Drop missing key fields
df_dys = df_dys.dropna(subset=['reader_view', 'speed'])

# Summary stats by reader_view
summary = df_dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std']).reset_index()

# t-test (Welch) on log-speed to reduce skew
# add small constant if any zeros (speed min seems >0)
log_speed = np.log(df_dys['speed'])
rv = df_dys['reader_view']
log_speed_0 = log_speed[rv == 0]
log_speed_1 = log_speed[rv == 1]

ttest_res = stats.ttest_ind(log_speed_1, log_speed_0, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d) on log-speed
mean_diff = log_speed_1.mean() - log_speed_0.mean()
# pooled SD for d (use unbiased pooled)
var1 = log_speed_1.var(ddof=1)
var0 = log_speed_0.var(ddof=1)
pooled_sd = np.sqrt(((len(log_speed_1)-1)*var1 + (len(log_speed_0)-1)*var0) / (len(log_speed_1)+len(log_speed_0)-2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Regression with controls: log(speed) ~ reader_view + page_id + num_words + device + age + gender + education + language + english_native + retake_trial
# Include fixed effects for page_id to control content; include participant random intercept if feasible.

# For robustness, use OLS with cluster-robust SE by uuid (if present)
reg_df = df_dys.copy()
reg_df = reg_df.dropna(subset=['speed', 'reader_view'])
reg_df['log_speed'] = np.log(reg_df['speed'])

# Build formula with available columns
controls = []
for col in ['page_id', 'num_words', 'device', 'age', 'gender', 'education', 'language', 'english_native', 'retake_trial', 'Flesch_Kincaid']:
    if col in reg_df.columns:
        if reg_df[col].dtype == 'object' or str(reg_df[col].dtype) == 'category':
            controls.append(f'C({col})')
        else:
            controls.append(col)

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join(controls)

model = smf.ols(formula, data=reg_df)
if 'uuid' in reg_df.columns:
    # cluster by participant; align groups to rows used by the model
    used_idx = model.data.row_labels
    groups = reg_df.loc[used_idx, 'uuid']
    res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
else:
    res = model.fit()

# Check within-participant variation of reader_view
if 'uuid' in df_dys.columns:
    rv_counts = df_dys.groupby('uuid')['reader_view'].nunique()
    both_conditions = (rv_counts > 1).sum()
    total_participants = rv_counts.shape[0]
else:
    both_conditions = np.nan
    total_participants = np.nan

output = {
    'n_rows_dyslexic': int(df_dys.shape[0]),
    'n_participants_dyslexic': int(df_dys['uuid'].nunique()) if 'uuid' in df_dys.columns else None,
    'both_conditions_participants': int(both_conditions) if both_conditions == both_conditions else None,
    'total_participants_with_rv': int(total_participants) if total_participants == total_participants else None,
    'summary_by_reader_view': summary.to_dict(orient='records'),
    'ttest_log_speed': {
        't_stat': float(ttest_res.statistic),
        'p_value': float(ttest_res.pvalue),
        'mean_log_speed_rv1': float(log_speed_1.mean()),
        'mean_log_speed_rv0': float(log_speed_0.mean()),
        'cohen_d_log': float(cohen_d),
    },
    'regression': {
        'formula': formula,
        'coef_reader_view': float(res.params.get('reader_view', np.nan)),
        'se_reader_view': float(res.bse.get('reader_view', np.nan)),
        'p_reader_view': float(res.pvalues.get('reader_view', np.nan)),
        'n_obs': int(res.nobs),
        'r2': float(res.rsquared),
    }
}

print(json.dumps(output, indent=2))
