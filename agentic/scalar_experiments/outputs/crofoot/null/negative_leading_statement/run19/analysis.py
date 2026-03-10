import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Derived variables
# Relative group size (focal - other)
df['rel_size'] = df['n_focal'] - df['n_other']
# Location advantage: positive if focal is closer to its own center than the other group is to its center
# (other distance minus focal distance)
df['loc_adv'] = df['dist_other'] - df['dist_focal']

# Logistic regression: win ~ rel_size + loc_adv
X = df[['rel_size', 'loc_adv']]
X = sm.add_constant(X)
y = df['win']

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Also model with separate distances to check robustness
X2 = df[['rel_size', 'dist_focal', 'dist_other']]
X2 = sm.add_constant(X2)
model2 = sm.GLM(y, X2, family=sm.families.Binomial())
result2 = model2.fit()

# Summaries
summary = {
    'n_rows': int(len(df)),
    'rel_size_mean': float(df['rel_size'].mean()),
    'loc_adv_mean': float(df['loc_adv'].mean()),
    'model1': {
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'conf_int': result.conf_int().to_dict(),
        'aic': float(result.aic),
        'pseudo_r2': float(1 - result.deviance / result.null_deviance),
    },
    'model2': {
        'params': result2.params.to_dict(),
        'pvalues': result2.pvalues.to_dict(),
        'conf_int': result2.conf_int().to_dict(),
        'aic': float(result2.aic),
        'pseudo_r2': float(1 - result2.deviance / result2.null_deviance),
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(result.summary())
print('\n---\n')
print(result2.summary())
