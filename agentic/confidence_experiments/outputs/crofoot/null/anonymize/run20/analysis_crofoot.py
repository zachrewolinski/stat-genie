import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Define variables
win = df['feature4']
rel_size = df['feature7'] - df['feature8']  # focal minus other
size_ratio = df['feature7'] / df['feature8']
rel_loc = df['feature6'] - df['feature5']   # positive means other farther from its center

# Standardize continuous predictors for comparable effects
X = pd.DataFrame({
    'rel_size': rel_size,
    'rel_loc': rel_loc,
})
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

# Logistic regression (GLM binomial)
model = sm.GLM(win, X_std, family=sm.families.Binomial())
res = model.fit()

# Alternative model using size ratio
X2 = pd.DataFrame({
    'size_ratio': size_ratio,
    'rel_loc': rel_loc,
})
X2_std = (X2 - X2.mean()) / X2.std(ddof=0)
X2_std = sm.add_constant(X2_std)
model2 = sm.GLM(win, X2_std, family=sm.families.Binomial())
res2 = model2.fit()

# Simple descriptive stats
win_rate = win.mean()
win_rate_bigger = win[rel_size > 0].mean()
win_rate_equal = win[rel_size == 0].mean() if (rel_size == 0).any() else np.nan
win_rate_smaller = win[rel_size < 0].mean()

# Location advantage: focal closer to its center => rel_loc > 0
win_rate_loc_adv = win[rel_loc > 0].mean()
win_rate_loc_disadv = win[rel_loc < 0].mean()

output = {
    'n': len(df),
    'win_rate': win_rate,
    'win_rate_bigger': win_rate_bigger,
    'win_rate_equal': win_rate_equal,
    'win_rate_smaller': win_rate_smaller,
    'win_rate_loc_adv': win_rate_loc_adv,
    'win_rate_loc_disadv': win_rate_loc_disadv,
    'glm_diff': {
        'params': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'conf_int': res.conf_int().to_dict(),
        'llf': res.llf,
        'aic': res.aic,
    },
    'glm_ratio': {
        'params': res2.params.to_dict(),
        'pvalues': res2.pvalues.to_dict(),
        'conf_int': res2.conf_int().to_dict(),
        'llf': res2.llf,
        'aic': res2.aic,
    }
}

print(json.dumps(output, indent=2))
