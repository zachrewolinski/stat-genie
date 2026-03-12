import json
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'teachingratings.csv'

df = pd.read_csv(DATA_PATH)

# Outcome and predictor
outcome = 'allstudents'
predictor = 'beauty'

# Basic correlation
pearson_r, pearson_p = stats.pearsonr(df[predictor], df[outcome])

# Simple OLS
model_simple = smf.ols(f"{outcome} ~ {predictor}", data=df).fit()
robust_simple = model_simple.get_robustcov_results(cov_type='HC3')

# Adjusted OLS with available covariates (excluding likely IDs)
# Treat categorical columns as categorical predictors
categoricals = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
# Only include if present
categoricals = [c for c in categoricals if c in df.columns]
continuous = [c for c in ['age', 'rownames', 'minority'] if c in df.columns]

terms = [predictor] + continuous + [f"C({c})" for c in categoricals]
formula = f"{outcome} ~ " + " + ".join(terms)
model_adj = smf.ols(formula, data=df).fit()
robust_adj = model_adj.get_robustcov_results(cov_type='HC3')

result = {
    'n': int(df.shape[0]),
    'pearson_r': float(pearson_r),
    'pearson_p': float(pearson_p),
    'simple_coef': float(model_simple.params[predictor]),
    'simple_p': float(robust_simple.pvalues[model_simple.params.index.get_loc(predictor)]),
    'simple_r2': float(model_simple.rsquared),
    'adj_coef': float(model_adj.params[predictor]),
    'adj_p': float(robust_adj.pvalues[model_adj.params.index.get_loc(predictor)]),
    'adj_r2': float(model_adj.rsquared),
    'formula_adj': formula,
}

print(json.dumps(result, indent=2))
