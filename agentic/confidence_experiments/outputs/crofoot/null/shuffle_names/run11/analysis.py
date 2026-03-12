import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Map variables based on info.json descriptions (names appear shuffled)
# Outcome: m_focal (binary 0/1)
# Relative group size: f_other (focal group size) vs win (other group size)
# Contest location: m_other (distance of focal from its home range center) vs n_focal (distance of other from its center)

outcome = df['m_focal']

focal_size = df['f_other']
other_size = df['win']
relative_size = focal_size - other_size

focal_dist = df['m_other']
other_dist = df['n_focal']
relative_location = other_dist - focal_dist  # positive means contest is closer to focal group's center

# Standardize predictors for comparability
X = pd.DataFrame({
    'rel_size': (relative_size - relative_size.mean()) / relative_size.std(ddof=0),
    'rel_location': (relative_location - relative_location.mean()) / relative_location.std(ddof=0),
})
X = sm.add_constant(X)

model = sm.Logit(outcome, X)
result = model.fit(disp=False)

# Also check single-predictor models for robustness
X_size = sm.add_constant(pd.DataFrame({
    'rel_size': (relative_size - relative_size.mean()) / relative_size.std(ddof=0)
}))
res_size = sm.Logit(outcome, X_size).fit(disp=False)

X_loc = sm.add_constant(pd.DataFrame({
    'rel_location': (relative_location - relative_location.mean()) / relative_location.std(ddof=0)
}))
res_loc = sm.Logit(outcome, X_loc).fit(disp=False)

# Basic descriptive stats
summary = {
    'n': int(df.shape[0]),
    'outcome_mean': float(outcome.mean()),
    'relative_size_mean': float(relative_size.mean()),
    'relative_size_std': float(relative_size.std(ddof=0)),
    'relative_location_mean': float(relative_location.mean()),
    'relative_location_std': float(relative_location.std(ddof=0)),
}

output = {
    'multivariable': {
        'coef': result.params.to_dict(),
        'pvalues': result.pvalues.to_dict(),
        'odds_ratio': np.exp(result.params).to_dict(),
        'conf_int': result.conf_int().rename(columns={0:'low',1:'high'}).to_dict('index'),
    },
    'size_only': {
        'coef': res_size.params.to_dict(),
        'pvalues': res_size.pvalues.to_dict(),
        'odds_ratio': np.exp(res_size.params).to_dict(),
        'conf_int': res_size.conf_int().rename(columns={0:'low',1:'high'}).to_dict('index'),
    },
    'location_only': {
        'coef': res_loc.params.to_dict(),
        'pvalues': res_loc.pvalues.to_dict(),
        'odds_ratio': np.exp(res_loc.params).to_dict(),
        'conf_int': res_loc.conf_int().rename(columns={0:'low',1:'high'}).to_dict('index'),
    },
    'summary': summary,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
