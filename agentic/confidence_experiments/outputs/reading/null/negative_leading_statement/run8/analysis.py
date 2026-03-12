import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Define dyslexia group (any dyslexia)
if 'dyslexia_bin' in df.columns:
    df['dyslexic'] = df['dyslexia_bin'] == 1
else:
    df['dyslexic'] = df['dyslexia'] > 0

# Clean speed (positive)
df = df.copy()
df = df[np.isfinite(df['speed'])]
df = df[df['speed'] > 0]

# Subset dyslexic participants
dd = df[df['dyslexic']]

# Group stats
stats_by_group = dd.groupby('reader_view')['speed'].agg(['count','mean','median','std']).reset_index()

# Two-sample Welch t-test
rv1 = dd[dd['reader_view']==1]['speed']
rv0 = dd[dd['reader_view']==0]['speed']

ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except ValueError:
    mwu = None

# Effect size (Cohen's d, Welch)
mean1, mean0 = rv1.mean(), rv0.mean()
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
# pooled SD with Welch weighting
n1, n0 = rv1.shape[0], rv0.shape[0]
pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2)) if (n1+n0-2)>0 else np.nan
cohens_d = (mean1-mean0)/pooled if pooled>0 else np.nan

# Regression on log(speed) with page controls
# Add small constant to avoid log(0) though speed>0
_dd = dd.copy()
_dd['log_speed'] = np.log(_dd['speed'])

# Use page_id or num_words if available
formula_terms = ['reader_view']
if 'page_id' in _dd.columns:
    formula_terms.append('C(page_id)')
if 'num_words' in _dd.columns:
    formula_terms.append('num_words')
if 'device' in _dd.columns:
    formula_terms.append('C(device)')

formula = 'log_speed ~ ' + ' + '.join(formula_terms)

model = smf.ols(formula, data=_dd).fit(cov_type='HC3')

# Extract reader_view coef
coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

output = {
    'n_total': int(len(df)),
    'n_dyslexic': int(len(dd)),
    'group_stats': stats_by_group.to_dict(orient='records'),
    'ttest_stat': float(ttest.statistic) if ttest is not None else None,
    'ttest_p': float(ttest.pvalue) if ttest is not None else None,
    'mwu_stat': float(mwu.statistic) if mwu is not None else None,
    'mwu_p': float(mwu.pvalue) if mwu is not None else None,
    'cohens_d': float(cohens_d),
    'reg_formula': formula,
    'reg_reader_view_coef': float(coef),
    'reg_reader_view_se': float(se),
    'reg_reader_view_p': float(pval),
}

with open('analysis_results.json','w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
