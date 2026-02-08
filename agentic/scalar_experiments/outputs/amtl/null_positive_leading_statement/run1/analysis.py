import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning and validation
_df = _df.copy()
_df = _df[_df['sockets'].notna() & _df['num_amtl'].notna()]
_df = _df[_df['sockets'] > 0]
_df = _df[_df['num_amtl'] >= 0]
_df = _df[_df['num_amtl'] <= _df['sockets']]

_df['failures'] = _df['sockets'] - _df['num_amtl']

# Fit binomial GLM with logit link
# Endog as 2-column success/failure for binomial counts
endog = _df[['num_amtl', 'failures']]

formula = 'C(genus) + age + prob_male + C(tooth_class)'
exog = patsy.dmatrix(formula, _df, return_type='dataframe')
design_info = exog.design_info
model = sm.GLM(endog, exog, family=sm.families.Binomial())
result = model.fit()

# Extract genus categories
design_matrix = patsy.dmatrix(formula, _df, return_type='dataframe')

# Function to get mean predicted AMTL rate for a given genus

def mean_predicted_rate(genus_value: str) -> float:
    data = _df.copy()
    data['genus'] = genus_value
    dm = patsy.build_design_matrices([design_info], data, return_type='dataframe')[0]
    pred = result.predict(dm)
    return float(np.average(pred, weights=data['sockets']))

# Unique genera
unique_genera = sorted(_df['genus'].unique())

# Predicted mean rates per genus
mean_rates = {g: mean_predicted_rate(g) for g in unique_genera}

# Wald tests for Homo sapiens vs each non-human genus
# Identify parameter names for genus
params = result.params.index.tolist()

# Identify baseline genus (the one not explicitly in parameters)
# With treatment coding, baseline is the first in alphabetical order by default
# Patsy usually chooses the first category alphabetically as baseline
# We'll infer it from design matrix columns

# Build contrast for Homo sapiens vs each other genus
wald_results = {}
for g in unique_genera:
    if g == 'Homo sapiens':
        continue
    # Construct a contrast vector for difference between Homo sapiens and g
    # Using the parameter names: C(genus)[T.<level>]
    # Baseline has no parameter.
    # Let beta_h = 0 if Homo sapiens is baseline, else beta_h is its coefficient.
    # Let beta_g = 0 if g is baseline, else beta_g is its coefficient.
    # We want beta_h - beta_g.
    L = np.zeros(len(result.params))
    name_h = f'C(genus)[T.Homo sapiens]'
    name_g = f'C(genus)[T.{g}]'
    if name_h in params:
        L[params.index(name_h)] = 1.0
    else:
        # Homo sapiens is baseline
        pass
    if name_g in params:
        L[params.index(name_g)] = L[params.index(name_g)] - 1.0
    else:
        # g is baseline
        L = L  # no change; subtracting 0
    test = result.t_test(L)
    wald_results[g] = {
        'effect_log_odds': float(test.effect),
        'se': float(test.sd),
        'z': float(test.tvalue),
        'pvalue': float(test.pvalue),
    }

# Print summary
print('N rows used:', len(_df))
print('Genera:', unique_genera)
print('\nGLM coefficients (log-odds):')
print(result.params)
print('\nPredicted mean AMTL rate by genus (weighted by sockets):')
for g, r in mean_rates.items():
    print(f'{g}: {r:.4f}')

print('\nWald tests: Homo sapiens vs each non-human genus (log-odds diff)')
for g, res in wald_results.items():
    print(f'Vs {g}: effect={res["effect_log_odds"]:.4f}, z={res["z"]:.3f}, p={res["pvalue"]:.4g}')
