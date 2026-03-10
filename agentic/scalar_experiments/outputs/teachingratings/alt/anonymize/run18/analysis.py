import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
DF = pd.read_csv('teachingratings.csv')

# Identify columns based on info.json mapping
# feature6: beauty rating (mean-centered)
# feature7: teaching evaluation score
beauty_col = 'feature6'
rating_col = 'feature7'

# Drop rows with missing values in key columns
key_cols = [beauty_col, rating_col]

# Simple correlation
corr_df = DF[key_cols].dropna()
pearson_r, pearson_p = stats.pearsonr(corr_df[beauty_col], corr_df[rating_col])

# Simple OLS
ols_simple = smf.ols(f"{rating_col} ~ {beauty_col}", data=DF).fit()

# Prepare multivariate regression with controls
# Categorical predictors: feature2 (minority), feature4 (gender), feature5 (single-credit),
# feature8 (upper/lower), feature9 (native english), feature10 (tenure track)
# Numeric controls: feature3 (age), feature11 (students participated), feature12 (students enrolled)
formula = (
    f"{rating_col} ~ {beauty_col} + C(feature2) + C(feature4) + C(feature5) + "
    f"C(feature8) + C(feature9) + C(feature10) + feature3 + feature11 + feature12"
)
ols_controls = smf.ols(formula, data=DF).fit()

# Standardized effect (beta) for beauty in controlled model
# Standardize numeric vars to get standardized coefficient
num_cols = [beauty_col, 'feature3', 'feature11', 'feature12']
std_df = DF.copy()
for c in num_cols:
    std_df[c] = (std_df[c] - std_df[c].mean()) / std_df[c].std(ddof=0)

ols_controls_std = smf.ols(formula, data=std_df).fit()

results = {
    'n_rows': int(len(DF)),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'ols_simple_coef': ols_simple.params.get(beauty_col, np.nan),
    'ols_simple_p': ols_simple.pvalues.get(beauty_col, np.nan),
    'ols_simple_r2': ols_simple.rsquared,
    'ols_controls_coef': ols_controls.params.get(beauty_col, np.nan),
    'ols_controls_p': ols_controls.pvalues.get(beauty_col, np.nan),
    'ols_controls_r2': ols_controls.rsquared,
    'ols_controls_std_beta': ols_controls_std.params.get(beauty_col, np.nan),
    'ols_controls_std_p': ols_controls_std.pvalues.get(beauty_col, np.nan),
}

print(json.dumps(results, indent=2))
