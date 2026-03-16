import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import json

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Map columns to readable names
col_map = {
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'division',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'students_eval',
    'feature12': 'students_enroll',
    'feature13': 'instructor_id',
}

df = df.rename(columns=col_map)

# Basic correlation
corr, corr_p = stats.pearsonr(df['beauty'], df['rating'])

# Simple OLS
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# Adjusted OLS with controls
formula = (
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
    'C(division) + C(native_english) + C(tenure_track) + students_eval + students_enroll'
)
model_adj = smf.ols(formula, data=df).fit()

# Cluster-robust SEs at instructor level
cluster_groups = df['instructor_id']
model_adj_cluster = model_adj.get_robustcov_results(cov_type='cluster', groups=cluster_groups)

# Extract stats for beauty
idx = list(model_adj.params.index).index('beauty')
coef = model_adj.params['beauty']
se = model_adj_cluster.bse[idx]
# t-stat and p-value with cluster-robust
# Use model_adj_cluster which already has robust bse and tvalues
p_value = model_adj_cluster.pvalues[idx]
ci = model_adj_cluster.conf_int()
ci_low, ci_high = [float(ci[idx][0]), float(ci[idx][1])]

# Standardized effect (per 1 SD of beauty)
beauty_sd = df['beauty'].std(ddof=1)
std_effect = coef * beauty_sd

# Save results for use in conclusion
results = {
    'n': len(df),
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'adj_coef': coef,
    'adj_se_cluster': se,
    'adj_p_cluster': p_value,
    'adj_ci_low': ci_low,
    'adj_ci_high': ci_high,
    'beauty_sd': beauty_sd,
    'std_effect': std_effect,
    'rating_mean': df['rating'].mean(),
    'rating_sd': df['rating'].std(ddof=1),
}

print(json.dumps(results, indent=2))
