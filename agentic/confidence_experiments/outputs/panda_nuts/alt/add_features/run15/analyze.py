import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)

# compute efficiency: nuts opened per second
# avoid division by zero; seconds minimum is >0 per metadata

df['efficiency'] = df['nuts_opened'] / df['seconds']

# encode predictors
# sex: f/m; help: y/N (note capital N)

df['sex'] = df['sex'].astype(str)
df['help'] = df['help'].astype(str)

# drop rows with missing values in key columns
model_df = df[['efficiency', 'age', 'sex', 'help', 'chimpanzee']].dropna().copy()

# create binary indicators
model_df['female'] = (model_df['sex'].str.lower() == 'f').astype(int)
model_df['help_yes'] = (model_df['help'].str.lower() == 'y').astype(int)

# Standardize age for interpretability (per SD) but also keep raw for effect size
model_df['age_z'] = (model_df['age'] - model_df['age'].mean()) / model_df['age'].std(ddof=0)

results = {}

# OLS with clustered SEs by chimpanzee
X = sm.add_constant(model_df[['age_z', 'female', 'help_yes']])
ols_model = sm.OLS(model_df['efficiency'], X).fit(cov_type='cluster', cov_kwds={'groups': model_df['chimpanzee']})
results['ols_cluster'] = {
    'params': ols_model.params.to_dict(),
    'pvalues': ols_model.pvalues.to_dict(),
    'r2': float(ols_model.rsquared),
    'n': int(ols_model.nobs),
}

# Mixed effects random intercept by chimpanzee (if converges)
# Use raw age to avoid issues with z in mixed
try:
    md = smf.mixedlm('efficiency ~ age_z + female + help_yes', model_df, groups=model_df['chimpanzee'])
    mdf = md.fit(reml=False)
    results['mixedlm'] = {
        'params': mdf.params.to_dict(),
        'pvalues': mdf.pvalues.to_dict(),
        'aic': float(mdf.aic),
        'bic': float(mdf.bic),
        'n': int(mdf.nobs),
        'converged': bool(mdf.converged),
    }
except Exception as e:
    results['mixedlm_error'] = str(e)

# Simple group means for interpretability
results['descriptives'] = {
    'efficiency_mean': float(model_df['efficiency'].mean()),
    'efficiency_std': float(model_df['efficiency'].std(ddof=0)),
    'age_mean': float(model_df['age'].mean()),
    'age_std': float(model_df['age'].std(ddof=0)),
    'n_rows': int(model_df.shape[0]),
    'n_chimps': int(model_df['chimpanzee'].nunique()),
    'female_rate': float(model_df['female'].mean()),
    'help_yes_rate': float(model_df['help_yes'].mean()),
}

# Means by help/sex
results['means_by_help'] = model_df.groupby('help_yes')['efficiency'].mean().to_dict()
results['means_by_sex'] = model_df.groupby('female')['efficiency'].mean().to_dict()

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
