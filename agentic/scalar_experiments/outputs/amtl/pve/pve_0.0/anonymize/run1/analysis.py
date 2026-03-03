import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Create human indicator
_df['human'] = (_df['feature8'] == 'Homo sapiens').astype(int)

# Fit OLS model controlling for age, sex, tooth class
# feature3: standardized AMTL count
model = smf.ols('feature3 ~ human + feature5 + feature7 + C(feature1)', data=_df).fit()

# Extract human effect
coef = model.params['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']

# Compute adjusted means for human vs non-human by counterfactual prediction
_df_h = _df.copy()
_df_h['human'] = 1
_df_n = _df.copy()
_df_n['human'] = 0
pred_h = model.predict(_df_h).mean()
pred_n = model.predict(_df_n).mean()

# Raw group means (for context)
raw_h = _df.loc[_df['human'] == 1, 'feature3'].mean()
raw_n = _df.loc[_df['human'] == 0, 'feature3'].mean()

# Effect size in SD units (since feature3 standardized)
# Cohen's d (difference in means / pooled sd)
# Use raw group means and pooled sd of feature3
h_vals = _df.loc[_df['human'] == 1, 'feature3']
n_vals = _df.loc[_df['human'] == 0, 'feature3']
pooled_sd = np.sqrt(((h_vals.var(ddof=1) * (len(h_vals)-1)) + (n_vals.var(ddof=1) * (len(n_vals)-1))) / (len(h_vals)+len(n_vals)-2))
cohens_d = (raw_h - raw_n) / pooled_sd if pooled_sd > 0 else np.nan

# Save key results
results = {
    'n_total': int(len(_df)),
    'n_human': int((_df['human'] == 1).sum()),
    'n_nonhuman': int((_df['human'] == 0).sum()),
    'coef_human': float(coef),
    'pval_human': float(pval),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'adj_mean_human': float(pred_h),
    'adj_mean_nonhuman': float(pred_n),
    'raw_mean_human': float(raw_h),
    'raw_mean_nonhuman': float(raw_n),
    'cohens_d_raw': float(cohens_d),
    'r2': float(model.rsquared),
}

print(results)
