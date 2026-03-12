import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
file_path = "crofoot.csv"
df = pd.read_csv(file_path)

# Compute relative size and relative location
# Relative group size: difference in total group size (focal - other)
df['size_diff'] = df['n_focal'] - df['n_other']

# Relative location: contest is closer to focal home range if dist_focal < dist_other
# Use difference (other - focal), so positive means closer to focal
# (i.e., other is farther from its own center relative to focal)
df['loc_diff'] = df['dist_other'] - df['dist_focal']

# Logistic regression: win ~ size_diff + loc_diff
X = df[['size_diff', 'loc_diff']]
X = sm.add_constant(X)
y = df['win']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Odds ratios and p-values
params = result.params
pvalues = result.pvalues
odds_ratios = np.exp(params)

# Also fit univariate models for sanity
models_uni = {}
for var in ['size_diff', 'loc_diff']:
    X_uni = sm.add_constant(df[[var]])
    res_uni = sm.Logit(y, X_uni).fit(disp=False)
    models_uni[var] = res_uni

summary = {
    "n_rows": len(df),
    "size_diff_mean": float(df['size_diff'].mean()),
    "loc_diff_mean": float(df['loc_diff'].mean()),
    "logit_multivariate": {
        "coef": params.to_dict(),
        "pvalues": pvalues.to_dict(),
        "odds_ratios": odds_ratios.to_dict(),
        "pseudo_r2": float(result.prsquared),
        "aic": float(result.aic)
    },
    "logit_univariate": {
        "size_diff": {
            "coef": models_uni['size_diff'].params.to_dict(),
            "pvalues": models_uni['size_diff'].pvalues.to_dict(),
            "odds_ratios": np.exp(models_uni['size_diff'].params).to_dict(),
            "pseudo_r2": float(models_uni['size_diff'].prsquared),
            "aic": float(models_uni['size_diff'].aic)
        },
        "loc_diff": {
            "coef": models_uni['loc_diff'].params.to_dict(),
            "pvalues": models_uni['loc_diff'].pvalues.to_dict(),
            "odds_ratios": np.exp(models_uni['loc_diff'].params).to_dict(),
            "pseudo_r2": float(models_uni['loc_diff'].prsquared),
            "aic": float(models_uni['loc_diff'].aic)
        }
    }
}

print(json.dumps(summary, indent=2))
