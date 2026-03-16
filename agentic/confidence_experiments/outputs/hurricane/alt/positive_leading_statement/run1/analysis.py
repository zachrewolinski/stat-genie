import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    'masfem','min','gender_mf','category','alldeaths','ndam','elapsedyrs','masfem_mturk','wind','ndam15','year'
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Prepare outcome: fatalities
# Use log1p to handle zeros and skew

df['log_deaths'] = np.log1p(df['alldeaths'])

# Correlations
corr_pearson = df[['masfem','alldeaths']].corr(method='pearson').iloc[0,1]
corr_spearman = df[['masfem','alldeaths']].corr(method='spearman').iloc[0,1]

corr_pearson_log = df[['masfem','log_deaths']].corr(method='pearson').iloc[0,1]
corr_spearman_log = df[['masfem','log_deaths']].corr(method='spearman').iloc[0,1]

# Simple t-test between female vs male names
female = df[df['gender_mf'] == 1]['alldeaths']
male = df[df['gender_mf'] == 0]['alldeaths']
# Use nonparametric Mann-Whitney and t-test on log1p
mw = stats.mannwhitneyu(female, male, alternative='two-sided')
tt = stats.ttest_ind(np.log1p(female), np.log1p(male), equal_var=False, nan_policy='omit')

# Regression specs
specs = {
    'model1_simple': ['masfem'],
    'model2_intensity': ['masfem', 'wind', 'min', 'category'],
    'model3_intensity_year': ['masfem', 'wind', 'min', 'category', 'year'],
    'model4_intensity_year_damage': ['masfem', 'wind', 'min', 'category', 'year', 'ndam15'],
}

reg_results = {}
for name, predictors in specs.items():
    data = df[['log_deaths'] + predictors].dropna()
    y = data['log_deaths']
    X = data[predictors]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    coef = model.params.get('masfem', np.nan)
    pval = model.pvalues.get('masfem', np.nan)
    conf = model.conf_int().loc['masfem'].tolist() if 'masfem' in model.params else [np.nan, np.nan]
    reg_results[name] = {
        'n': int(model.nobs),
        'coef_masfem': float(coef),
        'pval_masfem': float(pval),
        'conf_int_low': float(conf[0]),
        'conf_int_high': float(conf[1]),
        'r2': float(model.rsquared),
    }

# Also regression using masfem_mturk as robustness
specs_mturk = {
    'mturk_simple': ['masfem_mturk'],
    'mturk_intensity': ['masfem_mturk', 'wind', 'min', 'category'],
    'mturk_intensity_year': ['masfem_mturk', 'wind', 'min', 'category', 'year'],
    'mturk_intensity_year_damage': ['masfem_mturk', 'wind', 'min', 'category', 'year', 'ndam15'],
}

reg_results_mturk = {}
for name, predictors in specs_mturk.items():
    data = df[['log_deaths'] + predictors].dropna()
    y = data['log_deaths']
    X = data[predictors]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    coef = model.params.get('masfem_mturk', np.nan)
    pval = model.pvalues.get('masfem_mturk', np.nan)
    conf = model.conf_int().loc['masfem_mturk'].tolist() if 'masfem_mturk' in model.params else [np.nan, np.nan]
    reg_results_mturk[name] = {
        'n': int(model.nobs),
        'coef_masfem_mturk': float(coef),
        'pval_masfem_mturk': float(pval),
        'conf_int_low': float(conf[0]),
        'conf_int_high': float(conf[1]),
        'r2': float(model.rsquared),
    }

# Summaries
summary = {
    'n': int(df.shape[0]),
    'corr_pearson_alldeaths': float(corr_pearson),
    'corr_spearman_alldeaths': float(corr_spearman),
    'corr_pearson_logdeaths': float(corr_pearson_log),
    'corr_spearman_logdeaths': float(corr_spearman_log),
    'female_mean_deaths': float(female.mean()),
    'male_mean_deaths': float(male.mean()),
    'female_median_deaths': float(female.median()),
    'male_median_deaths': float(male.median()),
    'mw_stat': float(mw.statistic),
    'mw_p': float(mw.pvalue),
    'tt_log_stat': float(tt.statistic),
    'tt_log_p': float(tt.pvalue),
}

result = {
    'summary': summary,
    'reg_results_masfem': reg_results,
    'reg_results_masfem_mturk': reg_results_mturk,
}

with open('analysis_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
