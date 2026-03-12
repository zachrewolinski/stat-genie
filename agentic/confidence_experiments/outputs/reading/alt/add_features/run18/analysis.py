import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path('reading.csv')

df = pd.read_csv(DATA_PATH)

# Basic cleaning: ensure numeric columns are numeric
for col in ['speed', 'reader_view', 'dyslexia', 'dyslexia_bin']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Define dyslexia indicator: dyslexia_bin if available, else dyslexia > 0
if 'dyslexia_bin' in df.columns:
    df['dyslexic'] = df['dyslexia_bin'] == 1
else:
    df['dyslexic'] = df['dyslexia'] > 0

# Filter to dyslexic participants and valid speed values
sub = df.loc[df['dyslexic']].copy()
sub = sub.loc[sub['speed'].notna() & (sub['speed'] > 0)]
sub = sub.loc[sub['reader_view'].isin([0, 1])]

# Log-transform speed to reduce skew
sub['log_speed'] = np.log(sub['speed'])

# Summary stats
summary = sub.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Mixed effects model with random intercept by participant
# Fallback to OLS if uuid missing or model fails
model_result = None
model_type = None

if 'uuid' in sub.columns and sub['uuid'].notna().any():
    try:
        model = smf.mixedlm('log_speed ~ reader_view', sub, groups=sub['uuid'])
        model_result = model.fit(reml=False)
        model_type = 'mixedlm'
    except Exception:
        model_result = None

if model_result is None:
    model = smf.ols('log_speed ~ reader_view', data=sub)
    model_result = model.fit()
    model_type = 'ols'

# Extract effect
coef = model_result.params.get('reader_view', np.nan)
pval = model_result.pvalues.get('reader_view', np.nan)

# Convert log effect to percent change
percent_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

out = {
    'n_total': int(len(sub)),
    'n_reader_view_0': int((sub['reader_view'] == 0).sum()),
    'n_reader_view_1': int((sub['reader_view'] == 1).sum()),
    'summary_by_reader_view': summary.reset_index().to_dict(orient='records'),
    'model_type': model_type,
    'coef_log_speed_reader_view': float(coef),
    'p_value_reader_view': float(pval),
    'percent_change_speed': float(percent_change),
}

print(json.dumps(out, indent=2))
