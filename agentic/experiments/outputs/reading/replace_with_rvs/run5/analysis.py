import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = "reading.csv"
df = pd.read_csv(csv_path)

# Focus on individuals with dyslexia
if 'dyslexia_bin' in df.columns:
    # If dyslexia_bin isn't strictly binary, treat >= 0.5 as dyslexia
    unique_vals = set(pd.Series(df['dyslexia_bin'].dropna().unique()))
    if unique_vals.issubset({0, 1}):
        dyslexia_indicator = df['dyslexia_bin'] == 1
    else:
        dyslexia_indicator = df['dyslexia_bin'] >= 0.5
    dfx = df[dyslexia_indicator].copy()
else:
    # Fall back to dyslexia scale; treat >= 1 as dyslexia
    dfx = df[df['dyslexia'] >= 1].copy()

# Basic cleaning
for col in ['speed', 'reader_view']:
    dfx = dfx[np.isfinite(dfx[col])]

# Guard: require both groups present
rv_groups = dfx.groupby('reader_view')['speed']

summary = rv_groups.agg(['count', 'mean', 'median', 'std'])

# Two-sample t-test (Welch)
if set(dfx['reader_view'].unique()) >= {0, 1}:
    speed_rv1 = dfx[dfx['reader_view'] == 1]['speed']
    speed_rv0 = dfx[dfx['reader_view'] == 0]['speed']
    t_stat, p_val = stats.ttest_ind(speed_rv1, speed_rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

# Regression with controls on log speed to reduce skew
# Add small constant to avoid log(0) though speed min should be >0

dfx = dfx[dfx['speed'] > 0].copy()
dfx['log_speed'] = np.log(dfx['speed'])

# Use a modest set of controls to avoid overfitting
controls = []
for c in ['page_id', 'num_words', 'language']:
    if c in dfx.columns:
        # Skip controls with only one level/value in the dyslexia subset
        if dfx[c].nunique(dropna=True) > 1:
            controls.append(c)

formula = 'log_speed ~ reader_view'
if controls:
    formula += ' + ' + ' + '.join([f'C({c})' if dfx[c].dtype == 'object' or str(dfx[c].dtype) == 'category' else c for c in controls])

if len(dfx) > 0 and dfx['reader_view'].nunique() > 1:
    model = smf.ols(formula, data=dfx).fit(cov_type='HC3')
    coef = model.params.get('reader_view', np.nan)
    p_reg = model.pvalues.get('reader_view', np.nan)
else:
    coef = np.nan
    p_reg = np.nan

# Save analysis outputs for reference
summary.to_csv('analysis_summary.csv')

with open('analysis_results.txt', 'w') as f:
    f.write('Dyslexia-only summary by reader_view\n')
    f.write(summary.to_string())
    f.write('\n\n')
    f.write(f'Welch t-test: t={t_stat:.4f}, p={p_val:.6g}\n')
    f.write(f'Regression (log_speed) coef={coef:.6f}, p={p_reg:.6g}\n')

# Decide answer
improves = bool((coef > 0) and (p_reg < 0.05))

with open('conclusion.txt', 'w') as f:
    f.write('Yes\n' if improves else 'No\n')
    # brief reasoning
    mean0 = summary.loc[0, 'mean'] if 0 in summary.index else np.nan
    mean1 = summary.loc[1, 'mean'] if 1 in summary.index else np.nan
    f.write(f"Among readers with dyslexia, mean speed was {mean1:.1f} with Reader View vs {mean0:.1f} without. ")
    if np.isfinite(p_reg):
        f.write(f"A log-speed regression with controls finds a reader_view effect of {coef:.3f} (p={p_reg:.3g}), so the evidence does {'not ' if not improves else ''}support faster reading with Reader View.")
    else:
        f.write("Regression results were unavailable due to missing groups.")
