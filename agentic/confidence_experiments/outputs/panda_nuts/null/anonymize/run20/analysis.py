import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DATA_PATH = 'panda_nuts.csv'

df = pd.read_csv(DATA_PATH)

# Rename columns for clarity
col_map = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}

df = df.rename(columns=col_map)

# Basic cleaning
# Ensure categorical types
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Avoid zero/negative duration
if (df['duration_sec'] <= 0).any():
    raise ValueError('Non-positive duration found')

# Efficiency metrics
# nuts per minute for interpretability
# Also keep per-second rate for modeling via offset

df['nuts_per_min'] = df['nuts_opened'] / (df['duration_sec'] / 60.0)

# Poisson/NB GLM with offset for time to model counts, representing rate effects
# Formula includes age (continuous), sex, help
formula = 'nuts_opened ~ age + sex + help'

# Fit Poisson GLM
poisson_model = smf.glm(formula=formula, data=df,
                        family=sm.families.Poisson(),
                        offset=np.log(df['duration_sec']))
poisson_res = poisson_model.fit()

# Check overdispersion (deviance/df_resid)
overdispersion = poisson_res.deviance / poisson_res.df_resid

# Fit Negative Binomial if overdispersion is notable (>1.5)
nb_res = None
nb2_res = None
if overdispersion > 1.5:
    nb_model = smf.glm(formula=formula, data=df,
                       family=sm.families.NegativeBinomial(alpha=1.0),
                       offset=np.log(df['duration_sec']))
    nb_res = nb_model.fit()

    # Also fit NB2 with estimated overdispersion parameter
    nb2_model = smf.negativebinomial(formula=formula, data=df,
                                     offset=np.log(df['duration_sec']))
    nb2_res = nb2_model.fit(disp=False)

# Likelihood ratio test for joint effect vs intercept-only
null_model = smf.glm(formula='nuts_opened ~ 1', data=df,
                     family=sm.families.Poisson(),
                     offset=np.log(df['duration_sec']))
null_res = null_model.fit()

lr_stat = 2 * (poisson_res.llf - null_res.llf)
lr_df = poisson_res.df_model
lr_p = stats.chi2.sf(lr_stat, lr_df)

# Collect key results
results = {
    'n': int(df.shape[0]),
    'overdispersion': float(overdispersion),
    'poisson_params': poisson_res.params.to_dict(),
    'poisson_pvalues': poisson_res.pvalues.to_dict(),
    'lr_stat': float(lr_stat),
    'lr_df': float(lr_df),
    'lr_p': float(lr_p),
}

if nb_res is not None:
    results['nb_params'] = nb_res.params.to_dict()
    results['nb_pvalues'] = nb_res.pvalues.to_dict()
if nb2_res is not None:
    results['nb2_params'] = nb2_res.params.to_dict()
    results['nb2_pvalues'] = nb2_res.pvalues.to_dict()
    if 'alpha' in nb2_res.params.index:
        results['nb2_alpha'] = float(nb2_res.params['alpha'])

# Also compute simple group summaries for context
summary = {
    'nuts_per_min_mean': float(df['nuts_per_min'].mean()),
    'nuts_per_min_median': float(df['nuts_per_min'].median()),
    'nuts_per_min_by_sex': df.groupby('sex', observed=True)['nuts_per_min'].mean().to_dict(),
    'nuts_per_min_by_help': df.groupby('help', observed=True)['nuts_per_min'].mean().to_dict(),
}

results['summary'] = summary

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Wrote analysis_results.json')
