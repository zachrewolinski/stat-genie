import json
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Basic cleanup / derived variables
# Ensure categorical types are treated as such in the formula
# Create binary indicator for modern humans

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Model 1: human vs non-human with controls
model1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Model 2: add sockets as a control (robustness)
model2 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class) + sockets', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['specimen']}
)

# Effect size context
response_std = df['num_amtl'].std()

# Average marginal difference between humans and non-humans using model1
# Set is_human=1 and 0 for each row and average predicted difference
pred_human = model1.predict(df.assign(is_human=1))
pred_nonhuman = model1.predict(df.assign(is_human=0))
avg_diff = (pred_human - pred_nonhuman).mean()

results = {
    'n_rows': int(df.shape[0]),
    'n_specimens': int(df['specimen'].nunique()),
    'model1': {
        'coef_is_human': float(model1.params['is_human']),
        'se_is_human': float(model1.bse['is_human']),
        'p_is_human': float(model1.pvalues['is_human']),
        'ci_low': float(model1.conf_int().loc['is_human'][0]),
        'ci_high': float(model1.conf_int().loc['is_human'][1]),
        'avg_pred_diff': float(avg_diff),
    },
    'model2': {
        'coef_is_human': float(model2.params['is_human']),
        'se_is_human': float(model2.bse['is_human']),
        'p_is_human': float(model2.pvalues['is_human']),
        'ci_low': float(model2.conf_int().loc['is_human'][0]),
        'ci_high': float(model2.conf_int().loc['is_human'][1]),
    },
    'response_std': float(response_std),
    'mean_num_amtl_human': float(df.loc[df['is_human'] == 1, 'num_amtl'].mean()),
    'mean_num_amtl_nonhuman': float(df.loc[df['is_human'] == 0, 'num_amtl'].mean()),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
