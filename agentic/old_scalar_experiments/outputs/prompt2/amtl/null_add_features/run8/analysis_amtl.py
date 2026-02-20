import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# Load metadata (for potential future use or logging)
info_path = Path('info.json')
if info_path.exists():
    with info_path.open() as f:
        info = json.load(f)
else:
    info = {}

# Load data
csv_path = Path('amtl.csv')
df = pd.read_csv(csv_path)

# Basic cleaning / checks
# Keep only rows with non-missing key fields
key_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
for col in key_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in data")

df = df.dropna(subset=key_cols).copy()

# Exclude any rows with non-positive sockets to avoid invalid proportions
df = df[df['sockets'] > 0].copy()

# Create proportion of missing teeth
df['prop_amtl'] = df['num_amtl'] / df['sockets']

# Define human vs non-human primate indicator
df['is_human'] = (df['genus'].str.contains('Homo sapiens')).astype(int)

# Restrict to genera of interest: humans and the three non-human genera listed
mask_genus = df['genus'].isin(['Homo sapiens', 'Pan', 'Pongo', 'Papio'])
df_model = df[mask_genus].copy()

# Encode categorical predictors
# Tooth class: Anterior / Posterior / Premolar
X_cat = pd.get_dummies(df_model[['tooth_class']], drop_first=True)

# Age (continuous) and sex proxy (probability male)
X_cont = df_model[['age', 'prob_male']]

# Human indicator
X_human = df_model[['is_human']]

X = pd.concat([X_human, X_cont, X_cat], axis=1)
X = sm.add_constant(X, has_constant='add')

# Response: AMTL counts with sockets as binomial denominators
# Specify endog as a two-column array: [successes, failures]
endog = np.column_stack(
    [df_model['num_amtl'].to_numpy(), (df_model['sockets'] - df_model['num_amtl']).to_numpy()]
)
exposure = df_model['sockets']

# Use GLM with binomial family on aggregated counts
model = sm.GLM(
    endog,
    X,
    family=sm.families.Binomial(),
)
result = model.fit()

# Extract human coefficient and its statistics
coef_human = result.params['is_human']
se_human = result.bse['is_human']
z_human = coef_human / se_human if se_human != 0 else np.nan
p_human = 2 * (1 - norm.cdf(abs(z_human))) if not np.isnan(z_human) else np.nan

# Aggregate observed AMTL rates by human vs non-human
grouped = df_model.groupby('is_human').agg(
    total_amtl=('num_amtl', 'sum'),
    total_sockets=('sockets', 'sum'),
)
grouped['obs_rate'] = grouped['total_amtl'] / grouped['total_sockets']

obs_rate_non_human = float(grouped.loc[0, 'obs_rate'])
obs_rate_human = float(grouped.loc[1, 'obs_rate'])

# Odds ratio for humans vs non-humans from the model
odds_ratio_human = float(np.exp(coef_human))

# Predicted probabilities for an \"average\" profile, toggling human status
base = X.mean()
base['const'] = 1.0

row_non_human = base.copy()
row_non_human['is_human'] = 0

row_human = base.copy()
row_human['is_human'] = 1

row_non_human = row_non_human[result.params.index]
row_human = row_human[result.params.index]

logit_non_human = float(np.dot(row_non_human, result.params))
logit_human = float(np.dot(row_human, result.params))

prob_non_human = 1 / (1 + np.exp(-logit_non_human))
prob_human = 1 / (1 + np.exp(-logit_human))

# Summarize comparison
summary = {
    'coef_human': float(coef_human),
    'se_human': float(se_human),
    'z_human': float(z_human),
    'p_human': float(p_human),
    'odds_ratio_human_vs_non_human': odds_ratio_human,
    'obs_rate_non_human': obs_rate_non_human,
    'obs_rate_human': obs_rate_human,
    'prob_non_human_ref': float(prob_non_human),
    'prob_human_ref': float(prob_human),
    'n_rows_used': int(df_model.shape[0]),
}

print(json.dumps(summary, indent=2))
