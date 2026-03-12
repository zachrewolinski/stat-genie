import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('crofoot.csv')

# Keep relevant columns
# win (0/1), n_focal, n_other, dist_focal, dist_other

# Create relative group size and relative location metrics
# Relative group size: log ratio and difference
# Relative location: difference in distance to home range centers (other - focal)

df = df.copy()

# Ensure numeric
for col in ['win','n_focal','n_other','dist_focal','dist_other']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Derived predictors
# difference in group size and log ratio to avoid division issues

df['size_diff'] = df['n_focal'] - df['n_other']
# add small constant to avoid log(0) though sizes min 5

df['size_log_ratio'] = np.log(df['n_focal'] / df['n_other'])

# relative location: positive means focal closer to its own center than other is to its center?
# Use dist_other - dist_focal so positive implies other farther from its center, i.e., contest closer to focal's center

df['loc_diff'] = df['dist_other'] - df['dist_focal']

# Drop rows with missing
model_df = df[['win','size_diff','size_log_ratio','loc_diff']].dropna()

# Logistic regression with size_diff and loc_diff
model1 = smf.glm('win ~ size_diff + loc_diff', data=model_df, family=sm.families.Binomial()).fit()

# Logistic regression with size_log_ratio and loc_diff
model2 = smf.glm('win ~ size_log_ratio + loc_diff', data=model_df, family=sm.families.Binomial()).fit()

# Add interaction to see if location moderates size effect
model3 = smf.glm('win ~ size_diff * loc_diff', data=model_df, family=sm.families.Binomial()).fit()

# Summaries for key metrics
summary = {
    'n': int(model_df.shape[0]),
    'model1_params': model1.params.to_dict(),
    'model1_pvalues': model1.pvalues.to_dict(),
    'model1_confint': model1.conf_int().rename(columns={0:'ci_low',1:'ci_high'}).to_dict('index'),
    'model2_params': model2.params.to_dict(),
    'model2_pvalues': model2.pvalues.to_dict(),
    'model2_confint': model2.conf_int().rename(columns={0:'ci_low',1:'ci_high'}).to_dict('index'),
    'model3_params': model3.params.to_dict(),
    'model3_pvalues': model3.pvalues.to_dict(),
}

# Also compute effect sizes as odds ratios for model1
or_model1 = {k: float(np.exp(v)) for k, v in model1.params.items()}
summary['model1_odds_ratio'] = or_model1

# Save summary to json-like text for inspection
import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Rows:', model_df.shape[0])
print('Model1 params:', model1.params)
print('Model1 pvalues:', model1.pvalues)
print('Model2 params:', model2.params)
print('Model2 pvalues:', model2.pvalues)
print('Model3 pvalues:', model3.pvalues)
