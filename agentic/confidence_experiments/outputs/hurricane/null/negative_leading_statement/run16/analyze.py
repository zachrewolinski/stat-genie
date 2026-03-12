import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic setup
_df['log_deaths'] = np.log1p(_df['alldeaths'])

print('rows', len(_df))

# Correlations
for col in ['masfem', 'masfem_mturk', 'gender_mf']:
    pearson = _df['log_deaths'].corr(_df[col])
    spearman = _df['log_deaths'].corr(_df[col], method='spearman')
    print(f'corr log_deaths vs {col}: pearson={pearson:.4f}, spearman={spearman:.4f}')

# OLS models on log_deaths
ols_formulas = {
    'masfem': 'log_deaths ~ masfem + wind + min + category',
    'masfem_mturk': 'log_deaths ~ masfem_mturk + wind + min + category',
    'gender_mf': 'log_deaths ~ gender_mf + wind + min + category',
}

for label, formula in ols_formulas.items():
    model = smf.ols(formula, data=_df).fit()
    print(f'\nOLS {label} model:')
    print(model.summary())

# GLM count models on alldeaths
# Use Poisson and Negative Binomial to check robustness.
for label, formula in {
    'masfem': 'alldeaths ~ masfem + wind + min + category',
    'masfem_mturk': 'alldeaths ~ masfem_mturk + wind + min + category',
    'gender_mf': 'alldeaths ~ gender_mf + wind + min + category',
}.items():
    glm_pois = smf.glm(formula, data=_df, family=sm.families.Poisson()).fit()
    print(f'\nGLM Poisson {label} model:')
    print(glm_pois.summary())

    glm_nb = smf.glm(formula, data=_df, family=sm.families.NegativeBinomial()).fit()
    print(f'\nGLM NegBin {label} model:')
    print(glm_nb.summary())
