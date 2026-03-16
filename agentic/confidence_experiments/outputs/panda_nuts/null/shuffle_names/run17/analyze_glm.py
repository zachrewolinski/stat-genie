import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns
# nuts opened count in 'help', duration in 'chimpanzee'
# sex in 'nuts_opened' (f/m)
# help received in 'seconds' (y/N)

df['help_received'] = df['seconds'].map({'y': 'yes', 'Y': 'yes', 'N': 'no', 'n': 'no'})
df['sex_chimp'] = df['nuts_opened'].map({'f': 'f', 'm': 'm'})

analysis_df = df.dropna(subset=['help', 'chimpanzee', 'age', 'help_received', 'sex_chimp']).copy()

# Poisson GLM with log(duration) offset
analysis_df['log_duration'] = np.log(analysis_df['chimpanzee'])

model = smf.glm('help ~ age + C(sex_chimp) + C(help_received)',
                data=analysis_df,
                family=sm.families.Poisson(),
                offset=analysis_df['log_duration']).fit()

results = {
    'n': int(analysis_df.shape[0]),
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'deviance': float(model.deviance),
    'pearson_chi2': float(model.pearson_chi2),
}

with open('analysis_glm_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
