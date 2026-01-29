import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Create human indicator
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Avoid division by zero; data should have sockets >= 1
_df = _df[_df['sockets'] > 0].copy()

# Response as proportion with binomial weights
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Fit binomial GLM controlling for age, sex (prob_male), and tooth class
model = smf.glm(
    'amtl_rate ~ human + age + prob_male + C(tooth_class)',
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df['sockets']
).fit()

# Average marginal predictions for human vs non-human
pred_df_human = _df.copy()
pred_df_human['human'] = 1
pred_df_nonhuman = _df.copy()
pred_df_nonhuman['human'] = 0

pred_human = model.predict(pred_df_human).mean()
pred_nonhuman = model.predict(pred_df_nonhuman).mean()

coef = model.params.get('human', np.nan)
pval = model.pvalues.get('human', np.nan)

results = {
    'coef_human': float(coef),
    'pval_human': float(pval),
    'pred_mean_human': float(pred_human),
    'pred_mean_nonhuman': float(pred_nonhuman),
    'pred_diff': float(pred_human - pred_nonhuman),
}

print('Model coefficient (human):', results['coef_human'])
print('P-value (human):', results['pval_human'])
print('Predicted mean AMTL rate (human):', results['pred_mean_human'])
print('Predicted mean AMTL rate (non-human):', results['pred_mean_nonhuman'])
print('Predicted difference (human - non-human):', results['pred_diff'])
