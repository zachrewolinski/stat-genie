import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('hurricane.csv')

# Derived variables
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Controls based on metadata
controls = ['wind', 'min', 'category', 'ndam15', 'year']
controls = [c for c in controls if c in _df.columns]

# Helper to print key stats

def print_corr(x, y, label):
    pearson = stats.pearsonr(_df[x], _df[y])
    spearman = stats.spearmanr(_df[x], _df[y])
    print(f"{label} Pearson r={pearson.statistic:.3f}, p={pearson.pvalue:.3f}; Spearman rho={spearman.statistic:.3f}, p={spearman.pvalue:.3f}")

print('N total', len(_df))

# Correlations
print('\nCorrelations with deaths')
for col in ['masfem', 'masfem_mturk', 'gender_mf']:
    if col in _df.columns:
        print_corr(col, 'alldeaths', f'{col} vs alldeaths')
        print_corr(col, 'log_deaths', f'{col} vs log_deaths')

# OLS with robust SE
formula = 'log_deaths ~ masfem'
for c in controls:
    formula += f' + {c}'
model = smf.ols(formula, data=_df).fit(cov_type='HC3')
print('\nOLS (robust HC3):', formula)
print(model.summary().tables[1])

# OLS with gender_mf
if 'gender_mf' in _df.columns:
    formula_g = 'log_deaths ~ gender_mf'
    for c in controls:
        formula_g += f' + {c}'
    model_g = smf.ols(formula_g, data=_df).fit(cov_type='HC3')
    print('\nOLS (robust HC3):', formula_g)
    print(model_g.summary().tables[1])

# Negative Binomial GLM
formula_nb = 'alldeaths ~ masfem'
for c in controls:
    formula_nb += f' + {c}'
print('\nNegBin GLM:', formula_nb)
try:
    nb = smf.glm(formula_nb, data=_df, family=sm.families.NegativeBinomial()).fit()
    print(nb.summary().tables[1])
except Exception as e:
    print('NegBin failed:', e)

# Poisson with overdispersion check
formula_p = formula_nb
print('\nPoisson GLM:', formula_p)
try:
    poisson = smf.glm(formula_p, data=_df, family=sm.families.Poisson()).fit()
    print(poisson.summary().tables[1])
    # Overdispersion ratio
    ratio = poisson.deviance / poisson.df_resid
    print(f'Poisson overdispersion ratio (deviance/df): {ratio:.2f}')
except Exception as e:
    print('Poisson failed:', e)

# Save key results for later use

