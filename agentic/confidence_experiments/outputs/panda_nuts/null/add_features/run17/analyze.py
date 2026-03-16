import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)

# Keep only relevant columns and drop rows with missing
cols = ['chimpanzee','age','sex','help','nuts_opened','seconds']
sub = df[cols].copy()
sub = sub.dropna()

# Normalize categorical values
sub['sex'] = sub['sex'].astype(str).str.strip().str.lower()
sub['help'] = sub['help'].astype(str).str.strip().str.lower()

# Efficiency: nuts per second
sub['efficiency'] = sub['nuts_opened'] / sub['seconds']

# Encode categorical for formula
# use 'm' as reference if present
sub['sex'] = sub['sex'].replace({'female':'f','male':'m'})

# Ensure help yes/no mapping
sub['help'] = sub['help'].replace({'y':'y','yes':'y','n':'n','no':'n','N':'n'})

# Drop any unexpected categories
sub = sub[sub['sex'].isin(['m','f']) & sub['help'].isin(['y','n'])]

# OLS with categorical terms
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=sub).fit(cov_type='HC3')

# Cluster-robust SEs by chimpanzee (if enough clusters)
cluster_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['chimpanzee']})

# Joint F-test for predictors (age, sex, help) excluding intercept
param_names = list(model.params.index)
constraints = []
if 'age' in param_names:
    constraints.append('age = 0')
for name in param_names:
    if name.startswith('C(sex)'):
        constraints.append(f'{name} = 0')
    if name.startswith('C(help)'):
        constraints.append(f'{name} = 0')
f_test = model.f_test(', '.join(constraints)) if constraints else None

# Simple correlations for age vs efficiency
age_corr = sub[['age','efficiency']].corr().iloc[0,1]

# Group means for sex/help
means_sex = sub.groupby('sex')['efficiency'].mean().to_dict()
means_help = sub.groupby('help')['efficiency'].mean().to_dict()

results = {
    'n': int(len(sub)),
    'efficiency_summary': {
        'mean': float(sub['efficiency'].mean()),
        'std': float(sub['efficiency'].std()),
        'min': float(sub['efficiency'].min()),
        'max': float(sub['efficiency'].max())
    },
    'age_corr': float(age_corr),
    'means_sex': means_sex,
    'means_help': means_help,
    'ols_hc3': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'r2': float(model.rsquared),
        'adj_r2': float(model.rsquared_adj)
    },
    'ols_cluster': {
        'params': cluster_model.params.to_dict(),
        'pvalues': cluster_model.pvalues.to_dict(),
        'r2': float(cluster_model.rsquared)
    },
    'joint_f_test': None if f_test is None else {
        'fvalue': float(np.asarray(f_test.fvalue).item()),
        'pvalue': float(np.asarray(f_test.pvalue).item())
    }
}

with open('analysis_results.json','w') as f:
    json.dump(results,f,indent=2)

print(json.dumps(results, indent=2))
