import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop rows with missing key values
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df[cols].copy()

df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus'])

# Create human indicator
# Homo sapiens vs non-human primates (Pan, Pongo, Papio)
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Ensure valid counts
# Filter any rows where sockets <= 0 or num_amtl outside [0, sockets]
df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

# Fit binomial GLM with counts (successes/failures)
# Control for age, prob_male, and tooth_class
exog = patsy.dmatrix('is_human + age + prob_male + C(tooth_class)', data=df, return_type='dataframe')
endog = np.column_stack([df['num_amtl'].values, (df['sockets'] - df['num_amtl']).values])
model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract coefficient and p-value for is_human
coef = res.params.get('is_human', np.nan)
se = res.bse.get('is_human', np.nan)
pval = res.pvalues.get('is_human', np.nan)

# Compute odds ratio and 95% CI
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se) if np.isfinite(se) else np.nan
ci_high = np.exp(coef + 1.96 * se) if np.isfinite(se) else np.nan

# Compute marginal predicted AMTL rate for human vs non-human at mean covariates and modal tooth class
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# modal tooth class
modal_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [modal_tooth, modal_tooth],
    'sockets': [1, 1],
    'num_amtl': [0, 0]
})

pred_exog = patsy.dmatrix('is_human + age + prob_male + C(tooth_class)', data=pred_df, return_type='dataframe')
pred_exog = pred_exog.reindex(columns=exog.columns, fill_value=0)
pred = res.predict(pred_exog)
nonhuman_rate, human_rate = pred[0], pred[1]
rate_diff = human_rate - nonhuman_rate

# Decide Likert scale score based on effect direction and strength
# Heuristic: use sign of coef and p-value with effect size (odds ratio)
# Map to -100..100
if not np.isfinite(coef):
    score = 0
else:
    # strength based on p-value
    if pval < 0.001:
        strength = 90
    elif pval < 0.01:
        strength = 75
    elif pval < 0.05:
        strength = 60
    elif pval < 0.1:
        strength = 40
    else:
        strength = 20

    # adjust by odds ratio magnitude
    # small effect (<1.1) -> reduce; large effect (>2) -> increase
    if or_val < 1.1 and or_val > 0.9:
        strength = max(10, strength - 30)
    elif or_val > 2 or or_val < 0.5:
        strength = min(100, strength + 10)

    score = strength if coef > 0 else -strength

# Save summary stats for inspection
summary = {
    'n_rows': int(df.shape[0]),
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'odds_ratio': float(or_val),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'pred_nonhuman_rate': float(nonhuman_rate),
    'pred_human_rate': float(human_rate),
    'pred_rate_diff': float(rate_diff),
    'score': int(score)
}

pd.Series(summary).to_csv('analysis_summary.csv')

# Write conclusion.txt with integer score only
with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(int(score)))

print(res.summary())
print('\nSummary saved to analysis_summary.csv')
print(summary)
