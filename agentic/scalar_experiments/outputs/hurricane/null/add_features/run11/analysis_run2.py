import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

_df = pd.read_csv('hurricane.csv')
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Define control sets
control_sets = {
    'basic': ['wind', 'min', 'category', 'year'],
    'with_damage': ['wind', 'min', 'category', 'ndam15', 'year'],
}

for k, v in control_sets.items():
    control_sets[k] = [c for c in v if c in _df.columns]

print('N total', len(_df))

# Correlations for masfem
print('\nCorrelations (masfem)')
pearson = stats.pearsonr(_df['masfem'], _df['log_deaths'])
spearman = stats.spearmanr(_df['masfem'], _df['log_deaths'])
print(f'Pearson r={pearson.statistic:.3f}, p={pearson.pvalue:.3f}; Spearman rho={spearman.statistic:.3f}, p={spearman.pvalue:.3f}')

# OLS with robust SE for each control set
for name, controls in control_sets.items():
    formula = 'log_deaths ~ masfem'
    for c in controls:
        formula += f' + {c}'
    model = smf.ols(formula, data=_df).fit(cov_type='HC3')
    print(f"\nOLS HC3 ({name}): {formula}")
    print(model.summary().tables[1])

# Negative Binomial using discrete model (estimates alpha)
for name, controls in control_sets.items():
    formula = 'alldeaths ~ masfem'
    for c in controls:
        formula += f' + {c}'
    print(f"\nNegativeBinomial (discrete, {name}): {formula}")
    try:
        model_nb = smf.negativebinomial(formula, data=_df).fit(disp=0)
        print(model_nb.summary().tables[1])
        print(f"alpha (dispersion) = {model_nb.params.get('alpha', np.nan):.3f}")
    except Exception as e:
        print('NB failed:', e)

# Alternative: gender_mf as predictor (basic controls)
controls = control_sets['basic']
formula_g = 'log_deaths ~ gender_mf'
for c in controls:
    formula_g += f' + {c}'
if 'gender_mf' in _df.columns:
    model_g = smf.ols(formula_g, data=_df).fit(cov_type='HC3')
    print(f"\nOLS HC3 gender_mf (basic): {formula_g}")
    print(model_g.summary().tables[1])

