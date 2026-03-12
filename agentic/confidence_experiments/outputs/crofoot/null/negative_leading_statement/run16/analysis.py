import json
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('crofoot.csv')

# Relative group size (positive means focal larger)
df['rel_size'] = df['n_focal'] - df['n_other']
# Relative contest location (positive means contest closer to focal home range center)
df['rel_location'] = df['dist_other'] - df['dist_focal']

# Prepare model
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)

y = df['win']

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

# Also fit single-predictor models for robustness
models = {}
for col in ['rel_size', 'rel_location']:
    Xc = sm.add_constant(df[[col]])
    m = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()
    models[col] = m

# Summaries
out = {
    'n': int(len(df)),
    'rel_size_summary': df['rel_size'].describe().to_dict(),
    'rel_location_summary': df['rel_location'].describe().to_dict(),
    'glm_both': {
        'params': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'conf_int': result.conf_int().rename(columns={0:'low',1:'high'}).to_dict(),
        'aic': result.aic,
    },
    'glm_single': {
        k: {
            'params': v.params.to_dict(),
            'pvalues': v.pvalues.to_dict(),
            'conf_int': v.conf_int().rename(columns={0:'low',1:'high'}).to_dict(),
            'aic': v.aic,
        }
        for k, v in models.items()
    }
}

with open('analysis_output.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
