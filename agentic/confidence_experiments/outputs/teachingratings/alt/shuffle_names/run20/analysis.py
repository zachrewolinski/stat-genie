import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic stats
n = len(df)

# Pearson correlation between beauty and allstudents
corr, corr_p = stats.pearsonr(df['beauty'], df['allstudents'])

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Full model with available controls (treat categorical as factors)
# Exclude obvious identifiers (division, students, rownames) to avoid overfitting/meaningless controls
# Include eval, tenure, prof, native, gender, credits as categorical; age, minority as numeric
formula = 'allstudents ~ beauty + age + minority + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits)'
model_full = smf.ols(formula, data=df).fit()

# Extract key statistics
simple_beta = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

full_beta = model_full.params['beauty']
full_p = model_full.pvalues['beauty']

# Effect size: predicted change for 1 SD beauty in simple model
beauty_sd = df['beauty'].std(ddof=0)
allstudents_sd = df['allstudents'].std(ddof=0)

# Standardized effect (beta) from simple model
std_beta_simple = simple_beta * beauty_sd / allstudents_sd

# Collect results
out = {
    'n': int(n),
    'corr': float(corr),
    'corr_p': float(corr_p),
    'simple_beta': float(simple_beta),
    'simple_p': float(simple_p),
    'simple_r2': float(model_simple.rsquared),
    'full_beta': float(full_beta),
    'full_p': float(full_p),
    'full_r2': float(model_full.rsquared),
    'beauty_sd': float(beauty_sd),
    'allstudents_sd': float(allstudents_sd),
    'std_beta_simple': float(std_beta_simple),
}

# Save results for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
