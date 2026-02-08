import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: ensure non-missing, valid sockets >0
_df = _df.copy()
_df = _df[_df['sockets'] > 0]

# Define genus indicator and categorical predictors
_df['genus'] = _df['genus'].astype('category')
_df['tooth_class'] = _df['tooth_class'].astype('category')

# Binomial GLM with counts
# Use formula with genus + age + prob_male + tooth_class
model = smf.glm(
    formula='num_amtl ~ C(genus) + age + prob_male + C(tooth_class)',
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets']
)
res = model.fit()

# Create standardized covariate values for marginal predictions
mean_age = _df['age'].mean()
mean_prob_male = _df['prob_male'].mean()

# Build prediction frame for each genus at mean covariates and each tooth_class
# Then average across tooth_class levels equally
levels_genus = list(_df['genus'].cat.categories)
levels_tooth = list(_df['tooth_class'].cat.categories)

rows = []
for g in levels_genus:
    for t in levels_tooth:
        rows.append({'genus': g, 'age': mean_age, 'prob_male': mean_prob_male, 'tooth_class': t})

pred_df = pd.DataFrame(rows)

pred = res.get_prediction(pred_df)
summary = pred.summary_frame()
pred_df['pred_rate'] = summary['mean']

# Average across tooth classes equally
avg_pred = pred_df.groupby('genus', as_index=False)['pred_rate'].mean()

# Compute Homo vs non-human average difference
homo_rate = float(avg_pred.loc[avg_pred['genus'] == 'Homo sapiens', 'pred_rate'])
non_human_rate = float(avg_pred.loc[avg_pred['genus'] != 'Homo sapiens', 'pred_rate'].mean())

# Compute effect size in log-odds comparing Homo vs non-human average by contrast
# Use linear predictor difference at mean covariates and average tooth_class
# We'll compute predicted logit for each genus and average across tooth_class, then contrast
pred_df_lin = pred_df.copy()
lin_pred = res.get_prediction(pred_df_lin, linear=True).summary_frame()
pred_df_lin['lin'] = lin_pred['mean']

avg_lin = pred_df_lin.groupby('genus', as_index=False)['lin'].mean()
homo_lin = float(avg_lin.loc[avg_lin['genus'] == 'Homo sapiens', 'lin'])
non_human_lin = float(avg_lin.loc[avg_lin['genus'] != 'Homo sapiens', 'lin'].mean())

logit_diff = homo_lin - non_human_lin

# Approximate standard error for contrast using covariance of coefficients
# Build design matrix for each genus+tooth_class and average to get contrast vector
# Then compute variance = c' V c
# Note: statsmodels provides cov_params

params = res.params
cov = res.cov_params()

# Build design rows using model design info
from patsy import dmatrix

design = dmatrix(res.model.data.design_info, pred_df_lin)
# Average design rows for Homo and non-human
homo_mask = pred_df_lin['genus'] == 'Homo sapiens'
non_mask = pred_df_lin['genus'] != 'Homo sapiens'

homo_mean_row = np.asarray(design[homo_mask].mean(axis=0)).ravel()
non_mean_row = np.asarray(design[non_mask].mean(axis=0)).ravel()

contrast = homo_mean_row - non_mean_row

var = float(contrast @ cov.values @ contrast)
se = np.sqrt(var) if var > 0 else np.nan
z = logit_diff / se if se and np.isfinite(se) and se > 0 else np.nan

# Two-sided p-value
from scipy import stats
p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Save key results for downstream
out = {
    'homo_rate': homo_rate,
    'non_human_rate': non_human_rate,
    'rate_diff': homo_rate - non_human_rate,
    'logit_diff': logit_diff,
    'z': z,
    'p': p,
}

pd.Series(out).to_csv('analysis_results.csv')
