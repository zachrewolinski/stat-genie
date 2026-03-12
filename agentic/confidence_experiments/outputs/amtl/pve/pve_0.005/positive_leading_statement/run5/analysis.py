import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Compute AMTL rate per socket
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Indicator for modern humans
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Model 1: OLS on rate with robust SEs
model_rate = smf.ols('amtl_rate ~ is_human + age + prob_male + C(tooth_class)', data=_df).fit(cov_type='HC3')

# Model 2: OLS on counts with sockets as exposure covariate
model_count = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets', data=_df).fit(cov_type='HC3')

# Extract key stats from rate model
coef = model_rate.params['is_human']
se = model_rate.bse['is_human']
pval = model_rate.pvalues['is_human']

# Compute adjusted means for human vs non-human at average covariates
avg_age = _df['age'].mean()
avg_male = _df['prob_male'].mean()
classes = _df['tooth_class'].unique()

pred_rows = []
for tc in classes:
    pred_rows.append({'is_human': 1, 'age': avg_age, 'prob_male': avg_male, 'tooth_class': tc})
    pred_rows.append({'is_human': 0, 'age': avg_age, 'prob_male': avg_male, 'tooth_class': tc})

pred_df = pd.DataFrame(pred_rows)
prop = _df['tooth_class'].value_counts(normalize=True)
weights = [prop[row['tooth_class']] for _, row in pred_df.iterrows()]

pred = model_rate.predict(pred_df)
human_preds = pred[::2]
nonhuman_preds = pred[1::2]

human_mean = np.average(human_preds, weights=weights[::2])
nonhuman_mean = np.average(nonhuman_preds, weights=weights[1::2])

summary = {
    'rate_model': {
        'coef_is_human': float(coef),
        'se_is_human': float(se),
        'pval_is_human': float(pval),
        'human_mean_rate': float(human_mean),
        'nonhuman_mean_rate': float(nonhuman_mean),
    },
    'count_model': {
        'coef_is_human': float(model_count.params['is_human']),
        'se_is_human': float(model_count.bse['is_human']),
        'pval_is_human': float(model_count.pvalues['is_human']),
    },
    'n': int(len(_df))
}

print(summary)
