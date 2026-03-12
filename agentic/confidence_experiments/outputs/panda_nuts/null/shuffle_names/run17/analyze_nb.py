import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns

df['help_received'] = df['seconds'].map({'y': 'yes', 'Y': 'yes', 'N': 'no', 'n': 'no'})
df['sex_chimp'] = df['nuts_opened'].map({'f': 'f', 'm': 'm'})

analysis_df = df.dropna(subset=['help', 'chimpanzee', 'age', 'help_received', 'sex_chimp']).copy()
analysis_df['log_duration'] = np.log(analysis_df['chimpanzee'])

# Design matrix with intercept
X = pd.get_dummies(analysis_df[['age', 'sex_chimp', 'help_received']], drop_first=True)
X = sm.add_constant(X)

# Negative Binomial regression with offset
model = sm.NegativeBinomial(analysis_df['help'], X, offset=analysis_df['log_duration'])
res = model.fit(disp=False)

results = {
    'n': int(analysis_df.shape[0]),
    'params': res.params.to_dict(),
    'pvalues': res.pvalues.to_dict(),
    'alpha': float(res.params.get('alpha', float('nan'))),
    'llf': float(res.llf),
}

with open('analysis_nb_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
