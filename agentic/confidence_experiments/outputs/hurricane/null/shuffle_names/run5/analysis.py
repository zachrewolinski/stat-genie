import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('hurricane.csv')

# Map columns to meanings based on ranges and metadata
# deaths
_df['deaths'] = _df['name']
# femininity ratings
_df['fem_index'] = _df['category']          # 1-11 scale
_df['fem_index_mturk'] = _df['ind']         # 1-11 scale
_df['female_binary'] = _df['masfem_mturk']  # 0/1
# intensity / controls
_df['wind_mph'] = _df['year']               # max wind speed
_df['min_pressure'] = _df['ndam15']         # min pressure
_df['saffir'] = _df['gender_mf']            # 1-5 category
_df['hurricane_year'] = _df['wind']         # year

# Outcome transforms
_df['log_deaths'] = np.log(_df['deaths'] + 1)

# Helper to fit OLS and extract coefficients

def fit_ols(formula, data):
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    return model


def fit_nb(formula, data):
    model = smf.glm(formula, data=data, family=sm.families.NegativeBinomial()).fit()
    return model


results = {}

# Simple bivariate
results['ols_fem_only'] = fit_ols('log_deaths ~ fem_index', _df)
results['ols_fem_mturk_only'] = fit_ols('log_deaths ~ fem_index_mturk', _df)
results['ols_female_binary_only'] = fit_ols('log_deaths ~ female_binary', _df)

# Controls
base_controls = 'wind_mph + min_pressure + saffir'
results['ols_fem_controls'] = fit_ols(f'log_deaths ~ fem_index + {base_controls}', _df)
results['ols_fem_mturk_controls'] = fit_ols(f'log_deaths ~ fem_index_mturk + {base_controls}', _df)
results['ols_female_binary_controls'] = fit_ols(f'log_deaths ~ female_binary + {base_controls}', _df)

# Controls + year trend
results['ols_fem_controls_year'] = fit_ols(f'log_deaths ~ fem_index + {base_controls} + hurricane_year', _df)

# Interaction with severity (wind speed)
results['ols_fem_interaction'] = fit_ols(f'log_deaths ~ fem_index * wind_mph + min_pressure + saffir + hurricane_year', _df)

# Negative binomial for counts
results['nb_fem_controls'] = fit_nb(f'deaths ~ fem_index + {base_controls} + hurricane_year', _df)

# Summaries
summary_rows = []
for key, model in results.items():
    if 'fem_index' in model.params.index:
        coef = model.params['fem_index']
        pval = model.pvalues['fem_index']
        summary_rows.append((key, 'fem_index', coef, pval))
    if 'fem_index_mturk' in model.params.index:
        coef = model.params['fem_index_mturk']
        pval = model.pvalues['fem_index_mturk']
        summary_rows.append((key, 'fem_index_mturk', coef, pval))
    if 'female_binary' in model.params.index:
        coef = model.params['female_binary']
        pval = model.pvalues['female_binary']
        summary_rows.append((key, 'female_binary', coef, pval))
    if 'fem_index:wind_mph' in model.params.index:
        coef = model.params['fem_index:wind_mph']
        pval = model.pvalues['fem_index:wind_mph']
        summary_rows.append((key, 'fem_index:wind_mph', coef, pval))

summary_df = pd.DataFrame(summary_rows, columns=['model', 'term', 'coef', 'pval'])
print(summary_df.to_string(index=False))

# Correlations
corr = _df[['deaths', 'fem_index', 'fem_index_mturk', 'female_binary']].corr()
print('\nCorrelation matrix (Pearson):')
print(corr)

# effect size: per SD increase in fem_index on log_deaths for controls model
model = results['ols_fem_controls']
if 'fem_index' in model.params.index:
    sd_fem = _df['fem_index'].std()
    effect = model.params['fem_index'] * sd_fem
    print('\nEffect on log(deaths+1) per 1 SD fem_index (controls model):', effect)
    # convert to percent change approx
    pct = (np.exp(effect) - 1) * 100
    print('Approx % change in deaths+1 per 1 SD fem_index:', pct)

# output R-squared for key models
print('\nR2 controls model:', results['ols_fem_controls'].rsquared)
print('R2 controls+year model:', results['ols_fem_controls_year'].rsquared)
