import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# Ensure expected columns
required = {'chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Create efficiency: nuts per second

df = df.copy()
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Normalize categorical

df['sex'] = df['sex'].astype(str).str.lower()
df['help'] = df['help'].astype(str).str.lower()

# OLS model on efficiency

ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Poisson GLM for counts with offset log(seconds)
# Use robust SE as well

glm_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds']),
).fit(cov_type='HC3')

# Negative Binomial model for counts with offset
# Using discrete NegativeBinomial with log link and exposure
# exposure is seconds; it uses log(exposure) internally

nb_model = smf.negativebinomial(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    exposure=df['seconds'],
).fit(disp=False)

# Overdispersion check for Poisson

pearson_chi2 = glm_model.pearson_chi2
df_resid = glm_model.df_resid
overdispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

# Summaries

print('N rows:', len(df))
print('\nEfficiency summary (nuts/sec):')
print(df['efficiency'].describe())

print('\nOLS (efficiency) coefficients (HC3):')
print(ols_model.summary().tables[1])

print('\nPoisson GLM (counts with offset) coefficients (HC3):')
print(glm_model.summary().tables[1])
print(f"Overdispersion (Pearson chi2 / df): {overdispersion:.2f}")

print('\nNegative Binomial coefficients:')
print(nb_model.summary().tables[1])

# Effect sizes for GLM (rate ratios)

params = glm_model.params
conf = glm_model.conf_int()
rate_ratios = np.exp(params)
conf_rr = np.exp(conf)
print('\nPoisson rate ratios (exp coef) with 95% CI:')
for term in params.index:
    rr = rate_ratios[term]
    lo, hi = conf_rr.loc[term]
    print(f"{term}: RR={rr:.3f} (95% CI {lo:.3f}, {hi:.3f})")

# Save key results for later use if needed

results = {
    'n': int(len(df)),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_sd': float(df['efficiency'].std()),
    'ols_params': ols_model.params.to_dict(),
    'ols_pvalues': ols_model.pvalues.to_dict(),
    'glm_params': glm_model.params.to_dict(),
    'glm_pvalues': glm_model.pvalues.to_dict(),
    'overdispersion': float(overdispersion),
    'nb_params': nb_model.params.to_dict(),
    'nb_pvalues': nb_model.pvalues.to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
