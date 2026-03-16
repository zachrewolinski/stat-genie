import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data

df = pd.read_csv('amtl.csv')

# Create human indicator

df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Basic derived rate (may be noisy; used for sensitivity)

df['rate'] = df['feature3'] / df['feature4']

# Main model: continuous AMTL proxy with socket count as covariate
formula_main = 'feature3 ~ is_human + feature5 + feature7 + C(feature1) + feature4'

model_main = smf.ols(formula_main, data=df)
res_main = model_main.fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

# Sensitivity model: rate outcome (no feature4 covariate)
formula_rate = 'rate ~ is_human + feature5 + feature7 + C(feature1)'
model_rate = smf.ols(formula_rate, data=df)
res_rate = model_rate.fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

# Extract stats

def summarize_result(res, label):
    coef = res.params.get('is_human', np.nan)
    se = res.bse.get('is_human', np.nan)
    p = res.pvalues.get('is_human', np.nan)
    return {
        'label': label,
        'coef': float(coef),
        'se': float(se),
        'p': float(p),
        'n': int(res.nobs)
    }

summary_main = summarize_result(res_main, 'main')
summary_rate = summarize_result(res_rate, 'rate')

# Compute predicted difference for main model at mean covariates
mean_age = df['feature5'].mean()
mean_sex = df['feature7'].mean()
mean_sockets = df['feature4'].mean()
# Use the most common tooth class for prediction
mode_tooth = df['feature1'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'feature5': [mean_age, mean_age],
    'feature7': [mean_sex, mean_sex],
    'feature4': [mean_sockets, mean_sockets],
    'feature1': [mode_tooth, mode_tooth]
})

preds = res_main.get_prediction(pred_df).summary_frame(alpha=0.05)

# Save a concise report
report = {
    'summary_main': summary_main,
    'summary_rate': summary_rate,
    'predicted_main': {
        'tooth_class': mode_tooth,
        'mean_age': float(mean_age),
        'mean_sex': float(mean_sex),
        'mean_sockets': float(mean_sockets),
        'pred_nonhuman': float(preds['mean'].iloc[0]),
        'pred_human': float(preds['mean'].iloc[1]),
        'diff_human_minus_nonhuman': float(preds['mean'].iloc[1] - preds['mean'].iloc[0]),
        'ci_low_diff': float(preds['mean_ci_lower'].iloc[1] - preds['mean_ci_upper'].iloc[0]),
        'ci_high_diff': float(preds['mean_ci_upper'].iloc[1] - preds['mean_ci_lower'].iloc[0])
    }
}

import json
with open('analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
