import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep only relevant columns and drop rows with missing age
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
df = _df[cols].copy()
df = df.dropna(subset=['age'])

# Ensure valid counts
if (df['num_amtl'] > df['sockets']).any():
    raise ValueError('num_amtl exceeds sockets in some rows')

# Indicator for modern humans

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Proportion response with binomial GLM; weight by number of trials (sockets)

df['amtl_rate'] = df['num_amtl'] / df['sockets']

model = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    var_weights=df['sockets'],
)
result = model.fit()

# Extract effect for humans
coef = result.params['is_human']
se = result.bse['is_human']
pval = result.pvalues['is_human']

# Odds ratio and 95% CI
or_human = float(np.exp(coef))
ci_low, ci_high = np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)

# Adjusted (marginal standardized) AMTL rates for humans vs non-humans

def adjusted_rate(is_human_value: int) -> float:
    df_temp = df.copy()
    df_temp['is_human'] = is_human_value
    # Predict probability per row, then average
    preds = result.predict(df_temp)
    return float(preds.mean())

adj_human = adjusted_rate(1)
adj_nonhuman = adjusted_rate(0)

print('Rows used:', len(df))
print('GLM summary:')
print(result.summary())
print('\nHuman effect (log-odds):', coef)
print('p-value:', pval)
print('Odds ratio:', or_human)
print('95% CI OR:', (float(ci_low), float(ci_high)))
print('Adjusted AMTL rate (human):', adj_human)
print('Adjusted AMTL rate (non-human):', adj_nonhuman)
print('Adjusted difference (human - non-human):', adj_human - adj_nonhuman)
