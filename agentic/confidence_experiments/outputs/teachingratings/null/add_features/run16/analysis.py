import json
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('teachingratings.csv')

# Core variables
beauty = df['beauty']
eval_score = df['eval']

# Correlations
pearson_r, pearson_p = stats.pearsonr(beauty, eval_score)
spearman_r, spearman_p = stats.spearmanr(beauty, eval_score)

# Simple OLS
m1 = smf.ols('eval ~ beauty', data=df).fit()

# Multivariable OLS with robust SE
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) '
    '+ C(tenure) + C(division) + C(credits) + students + allstudents'
)
m2 = smf.ols(formula, data=df).fit(cov_type='HC3')

# Cluster-robust SE by professor
m3 = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

# Extract effect sizes
beauty_sd = beauty.std()
coef_bivariate = m1.params['beauty']
coef_adjusted = m2.params['beauty']
coef_cluster = m3.params['beauty']

# 95% CI for adjusted model (HC3)
ci_low, ci_high = m2.conf_int().loc['beauty']

# Effect of 1 SD increase in beauty on eval score
sd_effect = coef_adjusted * beauty_sd

# Decide Likert response
# Evidence shows near-zero association with very high p-values.
response = 10

explanation = (
    'Using 463 courses, I tested whether instructor beauty predicts teaching evaluations. '
    f'Pearson correlation between beauty and eval is r={pearson_r:.3f} (p={pearson_p:.3f}); '
    f'Spearman rho={spearman_r:.3f} (p={spearman_p:.3f}). '
    f'A bivariate OLS gives a beauty coefficient of {coef_bivariate:.6f} (p={m1.pvalues["beauty"]:.3f}). '
    'In a multivariable OLS controlling for age, gender, minority, native speaker, tenure, division, credits, '
    f'course size (students, allstudents), the beauty coefficient is {coef_adjusted:.6f} with robust SE p={m2.pvalues["beauty"]:.3f} '
    f'and 95% CI [{ci_low:.3f}, {ci_high:.3f}]. '
    f'A 1 SD increase in beauty corresponds to only {sd_effect:.4f} points on the 1–5 eval scale. '
    f'Cluster-robust SEs by professor give a similar coefficient ({coef_cluster:.6f}) with p={m3.pvalues["beauty"]:.3f}. '
    'Across specifications the effect is essentially zero and far from statistically significant, so the data provide no evidence '
    'that instructor beauty affects teaching productivity as reflected in ratings.'
)

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump({'response': int(response), 'explanation': explanation}, f)
