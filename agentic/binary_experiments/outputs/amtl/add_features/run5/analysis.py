import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('amtl.csv')

# Keep only the relevant genera
keep_genera = ['Homo sapiens', 'Pan', 'Pongo', 'Papio']
df = _df[_df['genus'].isin(keep_genera)].copy()

# Basic cleaning
# Ensure sockets and num_amtl are valid
valid = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df[valid].copy()

# Response as proportion with binomial weights
# Using freq_weights = sockets is equivalent to modeling counts with binomial trials
# Add small jitter to age if any missing? We'll drop missing rows for model vars.
model_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df_model = df[model_cols].dropna().copy()
df_model['amtl_rate'] = df_model['num_amtl'] / df_model['sockets']

# Fit binomial regression: control for age, sex (prob_male), tooth_class
# Set Homo sapiens as reference level
formula = 'amtl_rate ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df_model,
    family=sm.families.Binomial(),
    freq_weights=df_model['sockets']
)
result = model.fit()

# Marginal standardization: predict AMTL rate for each genus using same covariate distribution
pred_means = {}
for genus in keep_genera:
    temp = df_model.copy()
    temp['genus'] = genus
    pred = result.predict(temp)
    pred_means[genus] = float(np.average(pred, weights=temp['sockets']))

# Extract genus coefficients relative to Homo sapiens
coef = result.params
pvals = result.pvalues

summary = {
    'n_obs': int(df_model.shape[0]),
    'genera_counts': df_model['genus'].value_counts().to_dict(),
    'pred_means': pred_means,
    'coef': coef.to_dict(),
    'pvals': pvals.to_dict(),
}

# Save a concise text summary
with open('analysis_summary.txt', 'w') as f:
    f.write('Model: Binomial GLM with logit link\n')
    f.write(f'Observations used: {summary["n_obs"]}\n')
    f.write(f'Genus counts: {summary["genera_counts"]}\n\n')
    f.write('Predicted mean AMTL rate (marginalized over covariates):\n')
    for g, v in pred_means.items():
        f.write(f'  {g}: {v:.4f}\n')
    f.write('\nGenus coefficients vs Homo sapiens (log-odds scale):\n')
    for g in ['Pan', 'Pongo', 'Papio']:
        key = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
        if key in coef:
            f.write(f'  {g}: coef={coef[key]:.4f}, p={pvals[key]:.4g}\n')

# Print summary to stdout for quick inspection
print(result.summary())
print('\nPredicted mean AMTL rate by genus (marginalized):')
for g, v in pred_means.items():
    print(f'{g}: {v:.4f}')
