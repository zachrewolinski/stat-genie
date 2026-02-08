import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'amtl.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: drop rows with missing critical values
needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=needed).copy()

# Create human indicator
# Genus values include 'Homo sapiens' and non-human primates.
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Compute rate with guard against zero sockets
# Sockets min is 2 per metadata; still safe.
df = df[df['sockets'] > 0].copy()
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Fit binomial GLM with weights = sockets
# Controls: age, prob_male, tooth_class
model = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)
res = model.fit()

# Extract coefficient and p-value for is_human
coef = res.params['is_human']
pval = res.pvalues['is_human']

# Also compute predicted mean difference between human and non-human at average covariates
# Use average age, prob_male, and most common tooth_class
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
mode_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [mode_tooth, mode_tooth],
})

pred = res.predict(pred_df)
nonhuman_pred, human_pred = pred.iloc[0], pred.iloc[1]

# Save a small results summary for manual inspection
with open('analysis_summary.txt', 'w') as f:
    f.write(res.summary().as_text())
    f.write('\n\n')
    f.write(f'is_human coef (log-odds): {coef}\n')
    f.write(f'is_human p-value: {pval}\n')
    f.write(f'Predicted AMTL rate nonhuman: {nonhuman_pred}\n')
    f.write(f'Predicted AMTL rate human: {human_pred}\n')
    f.write(f'Predicted difference (human - nonhuman): {human_pred - nonhuman_pred}\n')
