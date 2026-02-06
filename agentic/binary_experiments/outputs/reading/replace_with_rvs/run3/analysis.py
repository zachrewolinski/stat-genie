import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Define dyslexic participants
# If dyslexia_bin is binary, use ==1; if it's continuous (e.g., replaced with random values), use threshold 0.5
if 'dyslexia_bin' in df.columns and df['dyslexia_bin'].notna().any():
    uniq = df['dyslexia_bin'].dropna().nunique()
    if uniq <= 2:
        dys_df = df[df['dyslexia_bin'] == 1].copy()
    else:
        dys_df = df[df['dyslexia_bin'] >= 0.5].copy()
elif 'dyslexia' in df.columns and df['dyslexia'].notna().any():
    uniq = df['dyslexia'].dropna().nunique()
    if uniq <= 3:
        dys_df = df[df['dyslexia'] >= 1].copy()
    else:
        dys_df = df[df['dyslexia'] >= df['dyslexia'].median()].copy()
else:
    dys_df = df.copy()

# Basic cleaning: drop missing key fields
key_cols = ['speed', 'reader_view']
for col in key_cols:
    dys_df = dys_df[dys_df[col].notna()]

# Summary stats by reader_view
summary = dys_df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Two-sample t-test (Welch) on raw speed when both groups have data
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val, dfree = ttest_ind(rv1, rv0, usevar='unequal')
else:
    t_stat, p_val, dfree = np.nan, np.nan, np.nan

# Log-speed analysis to mitigate skew
# Remove non-positive speeds (shouldn't exist, but be safe)
log_df = dys_df[dys_df['speed'] > 0].copy()
log_df['log_speed'] = np.log(log_df['speed'])

# Regression controlling for page and device and text difficulty
# Use available covariates; only include if columns exist
covariates = []
for col in ['page_id', 'device', 'num_words', 'Flesch_Kincaid', 'age', 'education', 'gender', 'english_native']:
    if col in log_df.columns:
        series = log_df[col]
        if series.dropna().empty:
            continue
        if series.dtype == 'object' or str(series.dtype).startswith('category'):
            if series.dropna().nunique() > 1:
                covariates.append(col)
        else:
            if series.dropna().nunique() > 1:
                covariates.append(col)

# Build formula
formula_parts = ['log_speed ~ reader_view']
for col in covariates:
    if log_df[col].dtype == 'object' or str(log_df[col].dtype).startswith('category'):
        formula_parts.append(f'C({col})')
    else:
        formula_parts.append(col)

formula = ' + '.join(formula_parts)

# Fit regression only if data is sufficient and reader_view has variation
if not log_df.empty and log_df['reader_view'].nunique() > 1:
    model = smf.ols(formula, data=log_df).fit()
    # Extract reader_view effect
    rv_coef = model.params.get('reader_view', np.nan)
    rv_p = model.pvalues.get('reader_view', np.nan)
else:
    model = None
    rv_coef = np.nan
    rv_p = np.nan

# Save analysis results
with open('analysis_results.txt', 'w') as f:
    f.write('Dyslexic subset size: %d\n' % len(dys_df))
    f.write('\nSummary speed by reader_view (0=off,1=on):\n')
    f.write(summary.to_string())
    f.write('\n\nWelch t-test on speed (rv1 vs rv0):\n')
    f.write(f't={t_stat:.4f}, p={p_val:.6g}, df={dfree:.1f}\n')
    f.write('\nLog-speed regression controlling for covariates:\n')
    f.write('Formula: ' + formula + '\n')
    f.write(f'reader_view coef={rv_coef:.6f}, p={rv_p:.6g}\n')
