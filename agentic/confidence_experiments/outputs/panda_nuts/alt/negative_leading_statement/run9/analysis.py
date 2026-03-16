import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('panda_nuts.csv')

# Efficiency: nuts opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic summaries
summary = {
    'n': len(df),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_std': df['efficiency'].std(),
}

# Correlations with age
pearson_r, pearson_p = stats.pearsonr(df['age'], df['efficiency'])
spearman_rho, spearman_p = stats.spearmanr(df['age'], df['efficiency'])

# Group differences: sex and help
def group_test(col):
    groups = [g['efficiency'].values for _, g in df.groupby(col)]
    # Welch t-test for two groups
    if len(groups) == 2:
        t_stat, t_p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    else:
        t_stat = t_p = u_stat = u_p = np.nan
    return {
        't_stat': t_stat, 't_p': t_p, 'u_stat': u_stat, 'u_p': u_p
    }

sex_tests = group_test('sex')
help_tests = group_test('help')

# OLS models with robust SEs
model_basic = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
model_cluster = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['chimpanzee']}
)
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['chimpanzee']}
)

def coef_table(model):
    return model.summary2().tables[1]

results = {
    'summary': summary,
    'pearson': {'r': pearson_r, 'p': pearson_p},
    'spearman': {'rho': spearman_rho, 'p': spearman_p},
    'sex_tests': sex_tests,
    'help_tests': help_tests,
    'ols_basic': coef_table(model_basic).to_dict(),
    'ols_cluster': coef_table(model_cluster).to_dict(),
    'ols_hammer_cluster': coef_table(model_hammer).to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Wrote analysis_results.json')
