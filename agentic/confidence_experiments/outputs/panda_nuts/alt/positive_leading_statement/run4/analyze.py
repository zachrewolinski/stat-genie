import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Clean / standardize
# Make sure categorical variables are treated as such
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency: nuts opened per second
# Add a small epsilon to avoid division by zero (though seconds min is 2.5)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Quick checks
summary = {
    'n_rows': int(df.shape[0]),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_std': float(df['efficiency'].std()),
}

# OLS regression with age, sex, help predicting efficiency
# Use robust SEs (HC3) for safety
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Also run a sensitivity model controlling for hammer type (could affect cracking)
# Not asked, but used as robustness check
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=df).fit(cov_type='HC3')

# Extract p-values and coefficients
results = {
    'main': {
        'params': model.params.to_dict(),
        'pvalues': model.pvalues.to_dict(),
        'r2': float(model.rsquared),
        'adj_r2': float(model.rsquared_adj),
    },
    'with_hammer': {
        'params': model_hammer.params.to_dict(),
        'pvalues': model_hammer.pvalues.to_dict(),
        'r2': float(model_hammer.rsquared),
        'adj_r2': float(model_hammer.rsquared_adj),
    }
}

# Compute partial eta squared for each predictor in main model via ANOVA
try:
    import statsmodels.api as sm
    anova = sm.stats.anova_lm(model, typ=2)
    # partial eta squared = SS_effect / (SS_effect + SS_error)
    ss_error = anova.loc['Residual', 'sum_sq']
    pes = {}
    for idx, row in anova.iterrows():
        if idx == 'Residual':
            continue
        pes[idx] = float(row['sum_sq'] / (row['sum_sq'] + ss_error))
    results['main']['partial_eta_sq'] = pes
except Exception as e:
    results['main']['partial_eta_sq_error'] = str(e)

# Save results for inspection
with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)
