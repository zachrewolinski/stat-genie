import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = _df[cols].copy()

# Basic cleaning
# Drop rows with missing essentials
needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=needed)

# Ensure valid counts
# Keep only rows with sockets > 0 and 0 <= num_amtl <= sockets
mask = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df.loc[mask].copy()

# Create human indicator
# Note: genus values include 'Homo sapiens' per info.json
# Treat all other genera as non-human

df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Outcome as proportion with binomial trials
# Use GLM Binomial with freq_weights = sockets

df['prop_amtl'] = df['num_amtl'] / df['sockets']

# Fit model
# Use tooth_class as categorical, include age and prob_male
model = smf.glm(
    formula='prop_amtl ~ human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)

result = model.fit(cov_type='HC0')

# Extract human effect
coef = result.params.get('human', np.nan)
se = result.bse.get('human', np.nan)
pval = result.pvalues.get('human', np.nan)

# Odds ratio and 95% CI
or_human = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# Compute adjusted predicted probabilities at mean covariates
# for human=1 and human=0
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# Use most common tooth_class for a simple reference prediction
mode_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [mode_tooth, mode_tooth]
})

pred_probs = result.predict(pred_df)

# Print summary for inspection
print(result.summary())
print('\nHuman effect (log-odds):', coef)
print('Odds ratio:', or_human)
print('95% CI:', (ci_low, ci_high))
print('p-value:', pval)
print('Adjusted predicted AMTL proportion (non-human, human):', pred_probs.values)

# Save key outputs to a small text file for reference if needed
with open('analysis_results.txt', 'w') as f:
    f.write(f"n_rows={len(df)}\n")
    f.write(f"human_coef={coef}\n")
    f.write(f"human_or={or_human}\n")
    f.write(f"human_or_ci=({ci_low},{ci_high})\n")
    f.write(f"human_pval={pval}\n")
    f.write(f"pred_nonhuman={pred_probs.iloc[0]}\n")
    f.write(f"pred_human={pred_probs.iloc[1]}\n")
