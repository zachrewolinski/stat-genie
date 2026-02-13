import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing key fields, if any
cols_needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df.dropna(subset=cols_needed).copy()

# Total observable teeth for the tooth class: AMTL (missing) + present sockets
# We model the proportion of AMTL teeth with binomial weights.
df['n_teeth'] = df['num_amtl'] + df['sockets']

# Exclude rows with zero observable teeth (no information)
df = df[df['n_teeth'] > 0].copy()

# Proportion of AMTL
df['prop_amtl'] = df['num_amtl'] / df['n_teeth']

# Indicator for modern humans vs non-human primates
# In this dataset, humans are encoded as 'Homo sapiens'.
df['is_human'] = (df['genus'].astype(str) == 'Homo sapiens').astype(int)

# Center and scale age lightly to aid model stability (optional)
age_mean = df['age'].mean()
age_std = df['age'].std(ddof=0)
if age_std == 0:
    df['age_c'] = 0.0
else:
    df['age_c'] = (df['age'] - age_mean) / age_std

# prob_male is already between 0 and 1; we can use it directly

# Fit binomial logistic regression with tooth_class as categorical
formula = 'prop_amtl ~ is_human + age_c + prob_male + C(tooth_class)'
model = smf.glm(formula=formula,
                data=df,
                family=sm.families.Binomial(),
                freq_weights=df['n_teeth'])
result = model.fit()

# Extract key quantities for interpretation
coef_is_human = result.params['is_human']
se_is_human = result.bse['is_human']
pval_is_human = result.pvalues['is_human']
odds_ratio_human = float(np.exp(coef_is_human))

# Predicted probabilities for a "typical" anterior tooth
# Use mean-centered age = 0, mean prob_male, and tooth_class='Anterior'.
base_data = {
    'is_human': [0, 1],
    'age_c': [0.0, 0.0],
    'prob_male': [df['prob_male'].mean(), df['prob_male'].mean()],
    'tooth_class': ['Anterior', 'Anterior'],
}
base_df = pd.DataFrame(base_data)

pred = result.get_prediction(base_df)
pred_summary = pred.summary_frame(alpha=0.05)

# Extract predicted probabilities and CIs
p_nonhuman_hat = float(pred_summary['mean'].iloc[0])
lo_nonhuman = float(pred_summary['mean_ci_lower'].iloc[0])
hi_nonhuman = float(pred_summary['mean_ci_upper'].iloc[0])

p_human_hat = float(pred_summary['mean'].iloc[1])
lo_human = float(pred_summary['mean_ci_lower'].iloc[1])
hi_human = float(pred_summary['mean_ci_upper'].iloc[1])

# Difference in predicted AMTL probabilities
abs_diff = p_human_hat - p_nonhuman_hat

print('N rows used:', len(df))
print('Model formula:', formula)
print('Coef is_human:', coef_is_human)
print('SE is_human:', se_is_human)
print('p-value is_human:', pval_is_human)
print('Odds ratio (human vs non-human):', odds_ratio_human)
print('Pred non-human (Anterior, avg age/sex):', p_nonhuman_hat,
      '95% CI [', lo_nonhuman, ',', hi_nonhuman, ']')
print('Pred human   (Anterior, avg age/sex):', p_human_hat,
      '95% CI [', lo_human, ',', hi_human, ']')
print('Absolute difference (human - non-human):', abs_diff)
