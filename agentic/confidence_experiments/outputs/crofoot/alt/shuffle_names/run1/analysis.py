import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Outcome: focal group won contest (binary)
y = _df['m_focal']

# Predictors based on metadata descriptions (names are shuffled but descriptions are authoritative)
# f_other: number of individuals in focal group
# win: number of individuals in other group
size_focal = _df['f_other']
size_other = _df['win']
rel_size = size_focal - size_other

# m_other: distance of focal group from its home range center
# n_focal: distance of other group from its home range center
# Relative contest location: focal distance minus other distance
rel_dist = _df['m_other'] - _df['n_focal']

# Standardize predictors for comparability
X = pd.DataFrame({
    'rel_size': rel_size,
    'rel_dist': rel_dist,
})
X_std = (X - X.mean()) / X.std(ddof=0)
X_std = sm.add_constant(X_std)

# Fit logistic regression
model = sm.Logit(y, X_std)
res = model.fit(disp=False)

# Confidence intervals and odds ratios
conf = res.conf_int()
conf.columns = ['ci_low', 'ci_high']

odds = np.exp(res.params)
odds_ci = np.exp(conf)

# Predicted probabilities at mean and +/-1 SD for each predictor (holding other at mean)
base = X_std.copy()
base[['rel_size', 'rel_dist']] = 0.0

# +/- 1 SD changes
scenarios = pd.DataFrame({
    'const': 1.0,
    'rel_size': [-1, 0, 1, 0, 0],
    'rel_dist': [0, 0, 0, -1, 1],
}, index=['size_-1sd', 'size_mean', 'size_+1sd', 'dist_-1sd', 'dist_+1sd'])

pred_probs = res.predict(scenarios)

# Separate models for each predictor (standardized)
X_size = sm.add_constant(((rel_size - rel_size.mean()) / rel_size.std(ddof=0)))
res_size = sm.Logit(y, X_size).fit(disp=False)

X_dist = sm.add_constant(((rel_dist - rel_dist.mean()) / rel_dist.std(ddof=0)))
res_dist = sm.Logit(y, X_dist).fit(disp=False)

summary = {
    'n': len(_df),
    'win_rate': float(y.mean()),
    'coef_full': res.params.to_dict(),
    'p_full': res.pvalues.to_dict(),
    'odds_full': odds.to_dict(),
    'odds_ci_full': odds_ci.to_dict(orient='index'),
    'predicted_probs': pred_probs.to_dict(),
    'coef_size_only': res_size.params.to_dict(),
    'p_size_only': res_size.pvalues.to_dict(),
    'coef_dist_only': res_dist.params.to_dict(),
    'p_dist_only': res_dist.pvalues.to_dict(),
}

print(summary)
