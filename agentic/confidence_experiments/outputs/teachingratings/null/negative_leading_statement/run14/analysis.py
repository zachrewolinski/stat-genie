import json
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DF = pd.read_csv('teachingratings.csv')

# Basic cleanup: ensure expected columns
# Convert categorical columns to category dtype where appropriate
cat_cols = [
    'minority', 'gender', 'credits', 'division', 'native', 'tenure'
]
for c in cat_cols:
    if c in DF.columns:
        DF[c] = DF[c].astype('category')

# Key variables
beauty = DF['beauty']
eval_score = DF['eval']

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(beauty, eval_score)

# Spearman correlation (robust to non-normality)
spearman_r, spearman_p = stats.spearmanr(beauty, eval_score)

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=DF).fit(cov_type='HC3')

# OLS with common controls
# Use all available controls except identifiers and eval/beauty
controls = ['minority', 'age', 'gender', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents']
# Ensure columns exist
controls = [c for c in controls if c in DF.columns]
formula = 'eval ~ beauty + ' + ' + '.join(controls)
model_controls = smf.ols(formula, data=DF).fit(cov_type='HC3')

# Extract coefficients
simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

controls_coef = model_controls.params['beauty']
controls_p = model_controls.pvalues['beauty']

# Effect size: change in eval for 1 SD beauty
beauty_sd = DF['beauty'].std()
eval_sd = DF['eval'].std()
# Predicted eval change for 1 SD increase in beauty
pred_change = simple_coef * beauty_sd
# Standardized effect size (beta)
std_beta = simple_coef * beauty_sd / eval_sd

results = {
    'n': int(len(DF)),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'spearman_r': float(spearman_r),
    'spearman_p': float(spearman_p),
    'simple_coef': float(simple_coef),
    'simple_p': float(simple_p),
    'controls_coef': float(controls_coef),
    'controls_p': float(controls_p),
    'beauty_sd': float(beauty_sd),
    'eval_sd': float(eval_sd),
    'pred_change_1sd_beauty': float(pred_change),
    'std_beta': float(std_beta),
    'simple_r2': float(model_simple.rsquared),
    'controls_r2': float(model_controls.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
