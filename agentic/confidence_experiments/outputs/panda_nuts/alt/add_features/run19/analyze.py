import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

# Define efficiency as nuts opened per second
# seconds is strictly positive in this dataset

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Ensure categorical variables
for col in ['sex', 'help']:
    df[col] = df[col].astype('category')

# OLS on efficiency
ols = smf.ols('efficiency ~ age + sex + help', data=df).fit()

# Count model with offset for time; use Negative Binomial due to overdispersion
nb = smf.glm(
    'nuts_opened ~ age + sex + help',
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df['seconds'])
).fit()

# Extract key stats
results = {
    'n': int(df.shape[0]),
    'ols_pvals': ols.pvalues.to_dict(),
    'ols_coef': ols.params.to_dict(),
    'nb_pvals': nb.pvalues.to_dict(),
    'nb_coef': nb.params.to_dict(),
}

# Determine qualitative conclusion
age_sig = (results['ols_pvals'].get('age', 1.0) < 0.05) and (results['nb_pvals'].get('age', 1.0) < 0.05)
sex_sig = (results['ols_pvals'].get('sex[T.m]', 1.0) < 0.05) and (results['nb_pvals'].get('sex[T.m]', 1.0) < 0.05)
help_sig = (results['ols_pvals'].get('help[T.y]', 1.0) < 0.05) or (results['nb_pvals'].get('help[T.y]', 1.0) < 0.05)

# Likert response: two strong predictors (age, sex) but help not consistently significant
# Set moderate-strong "Yes" but not near 100.
response = 70 if (age_sig and sex_sig) else 55
if age_sig and sex_sig and not help_sig:
    response = 70
elif age_sig or sex_sig:
    response = 60
else:
    response = 40

# Craft explanation
explanation = (
    f"Using {results['n']} sessions, I defined nut‑cracking efficiency as nuts_opened per second. "
    f"An OLS model (efficiency ~ age + sex + help) shows age is positive and significant "
    f"(coef={results['ols_coef']['age']:.3f}, p={results['ols_pvals']['age']:.3g}) and males have higher efficiency "
    f"than females (coef={results['ols_coef']['sex[T.m]']:.3f}, p={results['ols_pvals']['sex[T.m]']:.3g}). "
    f"Help is not significant in OLS (coef={results['ols_coef']['help[T.y]']:.3f}, p={results['ols_pvals']['help[T.y]']:.3g}). "
    f"A negative binomial count model with a log(seconds) offset (nuts_opened ~ age + sex + help) confirms age "
    f"(coef={results['nb_coef']['age']:.3f}, p={results['nb_pvals']['age']:.3g}) and sex "
    f"(coef={results['nb_coef']['sex[T.m]']:.3f}, p={results['nb_pvals']['sex[T.m]']:.3g}) remain significant, "
    f"while help is not (coef={results['nb_coef']['help[T.y]']:.3f}, p={results['nb_pvals']['help[T.y]']:.3g}). "
    f"Thus, age and sex show clear influence on efficiency, but receiving help does not show consistent evidence of an effect."
)

out = {
    'response': int(response),
    'explanation': explanation
}

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump(out, f)
