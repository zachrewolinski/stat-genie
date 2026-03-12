import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = "reading.csv"
df = pd.read_csv(path)

# Inspect relationship between feature20 and computed speeds
# compute speed based on time on page (feature4) and time minus scrolling (feature5)
# words: feature7
# speed_wpm = 60000 * words / time_ms

speed4 = 60000 * df['feature7'] / df['feature4']
speed5 = 60000 * df['feature7'] / df['feature5']

corr4 = np.corrcoef(speed4, df['feature20'])[0,1]
corr5 = np.corrcoef(speed5, df['feature20'])[0,1]

# Choose reading speed variable as feature20 if it matches speed4 or speed5 closely
# If correlation high with one, use feature20. Otherwise use speed5.

# Identify dyslexia individuals
# feature17 binary dyslexia indicator; feature12 levels (0 no, 1 dyslexia, 2 severe)

# We'll primarily use feature17==1 as dyslexia, but also check feature12>0 as sensitivity

def analyze(subset, label):
    # Use feature20 as reading speed (wpm) if aligned with computed speed
    y = subset['feature20'].copy()
    # reader view: feature3 1 vs 0
    rv = subset['feature3']
    group1 = y[rv == 1]
    group0 = y[rv == 0]
    # Basic stats
    res = {}
    res['n_total'] = len(subset)
    res['n_rv1'] = len(group1)
    res['n_rv0'] = len(group0)
    res['mean_rv1'] = group1.mean()
    res['mean_rv0'] = group0.mean()
    res['median_rv1'] = group1.median()
    res['median_rv0'] = group0.median()
    res['diff_mean'] = res['mean_rv1'] - res['mean_rv0']
    res['diff_median'] = res['median_rv1'] - res['median_rv0']
    # Welch t-test
    tstat, pval = stats.ttest_ind(group1, group0, equal_var=False, nan_policy='omit')
    res['t_pvalue'] = pval
    # Mann-Whitney U (two-sided)
    try:
        ustat, pval_u = stats.mannwhitneyu(group1, group0, alternative='two-sided')
    except ValueError:
        pval_u = np.nan
    res['mw_pvalue'] = pval_u
    # Effect size: Cohen's d (Welch)
    # compute pooled SD with unequal n
    n1, n0 = group1.count(), group0.count()
    s1, s0 = group1.var(ddof=1), group0.var(ddof=1)
    pooled = np.sqrt((s1/n1 + s0/n0))
    res['cohen_d'] = (res['mean_rv1'] - res['mean_rv0']) / pooled if pooled > 0 else np.nan

    # Robust regression controlling for page (feature2) and words (feature7) maybe language and retake
    # Use log of speed to reduce outliers
    # Build model: log(speed) ~ reader_view + words + page fixed effects
    sub = subset.copy()
    sub = sub[np.isfinite(sub['feature20'])]
    sub = sub[sub['feature20'] > 0]
    sub['log_speed'] = np.log(sub['feature20'])
    # design matrix
    # include reader view, words, page (categorical), language, device, retake
    # to avoid too many categories, include page fixed effects only
    X = pd.get_dummies(sub[['feature3', 'feature7', 'feature2']], columns=['feature2'], drop_first=True)
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(sub['log_speed'], X).fit()
    res['reg_coef_rv'] = model.params.get('feature3', np.nan)
    res['reg_pvalue_rv'] = model.pvalues.get('feature3', np.nan)
    res['reg_n'] = int(model.nobs)
    return res

results = {}
results['corr_speed4_feature20'] = corr4
results['corr_speed5_feature20'] = corr5

# Dyslexia by feature17
results['dyslexia_feature17'] = analyze(df[df['feature17'] == 1], 'feature17')

# Dyslexia by feature12>0
results['dyslexia_feature12'] = analyze(df[df['feature12'] > 0], 'feature12')

# Also check non-dyslexia for context
results['non_dyslexia_feature17'] = analyze(df[df['feature17'] == 0], 'non')

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
