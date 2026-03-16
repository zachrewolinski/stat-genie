import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic info
n = len(df)

# Ensure types
# feature6 (beauty) and feature7 (teaching rating) numeric
beauty = pd.to_numeric(df['feature6'], errors='coerce')
rating = pd.to_numeric(df['feature7'], errors='coerce')

valid = beauty.notna() & rating.notna()
beauty = beauty[valid]
rating = rating[valid]

# Pearson correlation
r, p_r = stats.pearsonr(beauty, rating)

# Simple OLS
model_simple = smf.ols('feature7 ~ feature6', data=df).fit()

# Standardized coefficient (beta) for beauty
# z-score both variables
z_beauty = (beauty - beauty.mean()) / beauty.std(ddof=0)
z_rating = (rating - rating.mean()) / rating.std(ddof=0)
# slope in standardized units
model_std = smf.ols('z_rating ~ z_beauty', data=pd.DataFrame({'z_rating': z_rating, 'z_beauty': z_beauty})).fit()

# Multiple regression with controls (categoricals handled by patsy)
# feature2 minority, feature3 age, feature4 gender, feature5 single credit, feature8 upper/lower,
# feature9 native english, feature10 tenure track, feature11 students eval, feature12 enrolled
formula_controls = (
    'feature7 ~ feature6 + C(feature2) + feature3 + C(feature4) + C(feature5) '
    '+ C(feature8) + C(feature9) + C(feature10) + feature11 + feature12'
)
model_controls = smf.ols(formula_controls, data=df).fit()

result = {
    'n': int(n),
    'corr_r': float(r),
    'corr_p': float(p_r),
    'simple_coef': float(model_simple.params.get('feature6', np.nan)),
    'simple_p': float(model_simple.pvalues.get('feature6', np.nan)),
    'simple_ci': [float(x) for x in model_simple.conf_int().loc['feature6'].tolist()],
    'std_beta': float(model_std.params.get('z_beauty', np.nan)),
    'std_p': float(model_std.pvalues.get('z_beauty', np.nan)),
    'controls_coef': float(model_controls.params.get('feature6', np.nan)),
    'controls_p': float(model_controls.pvalues.get('feature6', np.nan)),
    'controls_ci': [float(x) for x in model_controls.conf_int().loc['feature6'].tolist()],
    'r2_simple': float(model_simple.rsquared),
    'r2_controls': float(model_controls.rsquared)
}

print(json.dumps(result, indent=2))
