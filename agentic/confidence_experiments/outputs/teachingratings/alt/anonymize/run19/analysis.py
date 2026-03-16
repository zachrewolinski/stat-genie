import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Map column names based on info.json descriptions
_df = _df.rename(
    columns={
        'feature1': 'course_id',
        'feature2': 'minority',
        'feature3': 'age',
        'feature4': 'gender',
        'feature5': 'single_credit',
        'feature6': 'beauty',
        'feature7': 'evaluation',
        'feature8': 'upper_div',
        'feature9': 'native_english',
        'feature10': 'tenure_track',
        'feature11': 'n_eval',
        'feature12': 'n_enroll',
        'feature13': 'instructor_id',
    }
)

# Basic Pearson correlation
corr, corr_p = stats.pearsonr(_df['beauty'], _df['evaluation'])

# OLS with controls; use robust SEs (HC3)
formula = (
    'evaluation ~ beauty + age + C(gender) + C(minority) + C(single_credit) + '
    'C(upper_div) + C(native_english) + C(tenure_track) + n_eval + n_enroll'
)
model = smf.ols(formula, data=_df).fit(cov_type='HC3')

beauty_coef = model.params['beauty']
beauty_se = model.bse['beauty']
beauty_p = model.pvalues['beauty']

# Standardized effect: per 1 SD increase in beauty, change in evaluation SDs
beauty_sd = _df['beauty'].std(ddof=1)
eval_sd = _df['evaluation'].std(ddof=1)
std_effect = beauty_coef * beauty_sd / eval_sd if eval_sd != 0 else np.nan

# Also compute 95% CI for beauty coefficient
ci_low, ci_high = model.conf_int().loc['beauty']

results = {
    'n': int(_df.shape[0]),
    'corr': float(corr),
    'corr_p': float(corr_p),
    'beauty_coef': float(beauty_coef),
    'beauty_se': float(beauty_se),
    'beauty_p': float(beauty_p),
    'beauty_ci_low': float(ci_low),
    'beauty_ci_high': float(ci_high),
    'std_effect': float(std_effect),
    'eval_sd': float(eval_sd),
    'beauty_sd': float(beauty_sd),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
