import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('affairs.csv')

# Map columns per info.json
# feature2: affair frequency; feature6: children yes/no

df['children'] = df['feature6'].str.strip().str.lower()

# Binary outcome: any affairs
if df['feature2'].dtype == object:
    df['affairs'] = pd.to_numeric(df['feature2'], errors='coerce')
else:
    df['affairs'] = df['feature2']

# remove missing
clean = df.dropna(subset=['affairs', 'children']).copy()

clean['any_affair'] = (clean['affairs'] > 0).astype(int)

# Group summaries
summary = clean.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any=('any_affair', 'mean')
)

# Mann-Whitney U test for affairs frequency (ordinal)
# Use two-sided; children categories assumed 'yes' and 'no'
children_vals = clean['children'].unique().tolist()
# Ensure order
if set(children_vals) >= {'yes','no'}:
    grp_yes = clean.loc[clean['children']=='yes', 'affairs']
    grp_no = clean.loc[clean['children']=='no', 'affairs']
else:
    # fallback: first two unique
    grp_yes = clean.loc[clean['children']==children_vals[0], 'affairs']
    grp_no = clean.loc[clean['children']==children_vals[1], 'affairs']

mw = stats.mannwhitneyu(grp_yes, grp_no, alternative='two-sided')

# Chi-square test for any affair
cont = pd.crosstab(clean['children'], clean['any_affair'])
chi2 = stats.chi2_contingency(cont)

# Logistic regression for any affair with controls
# Use statsmodels; encode categorical gender + children
# Other predictors: age (feature4), years married (feature5), religiousness (feature7), education (feature8), occupation (feature9), marriage rating (feature10)
# Drop missing rows
model_df = clean.dropna(subset=['feature3','feature4','feature5','feature7','feature8','feature9','feature10']).copy()

# Build model
formula = 'any_affair ~ C(children) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'

logit_model = smf.logit(formula, data=model_df).fit(disp=False)

# Extract odds ratio and p-value for children (yes vs no)
# By default, reference is first category in alphabetical order; need to check
params = logit_model.params
pvalues = logit_model.pvalues

# Find children term
child_terms = [term for term in params.index if term.startswith('C(children)')]

results = {
    'summary': summary.to_dict(),
    'mw_stat': mw.statistic,
    'mw_p': mw.pvalue,
    'chi2_stat': chi2[0],
    'chi2_p': chi2[1],
    'chi2_dof': chi2[2],
    'contingency': cont.to_dict(),
    'logit_params': params.to_dict(),
    'logit_pvalues': pvalues.to_dict(),
    'logit_child_terms': child_terms,
    'logit_n': int(logit_model.nobs)
}

print(json.dumps(results, indent=2))
