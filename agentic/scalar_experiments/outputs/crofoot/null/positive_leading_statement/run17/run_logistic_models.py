import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('crofoot.csv')

# Construct key predictors
# Relative group size (difference and ratio)
df['rel_size'] = df['n_focal'] - df['n_other']
# Avoid division by zero (not an issue here but keep safe)
df['size_ratio'] = df['n_focal'] / df['n_other']

# Location advantage: positive if focal is closer to its own home range center
# (smaller distance => closer; so dist_other - dist_focal > 0 means focal closer)
df['loc_advantage'] = df['dist_other'] - df['dist_focal']

print('Data shape:', df.shape)
print('Win counts:\n', df['win'].value_counts())

# Helper to run and print a simple logistic regression

def run_logit(name, formula_X_cols):
    print('\n' + '=' * 80)
    print(f'Model: {name}')
    X = df[formula_X_cols].copy()
    X = sm.add_constant(X)
    y = df['win']
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    print(result.summary())
    # Also print odds ratios and p-values
    params = result.params
    conf = result.conf_int()
    odds = params.map(lambda b: float(pd.np.exp(b)))  # type: ignore[attr-defined]
    conf_odds = conf.apply(lambda c: pd.np.exp(c), axis=1)  # type: ignore[attr-defined]
    print('\nOdds ratios with 95% CI and p-values:')
    for col in result.params.index:
        print(f"  {col:15s} OR={odds[col]:.3f}, CI=({conf_odds.loc[col, 0]:.3f}, {conf_odds.loc[col, 1]:.3f}), p={result.pvalues[col]:.3f}")
    return result

# 1) Model with relative size and location advantage
m1 = run_logit('win ~ rel_size + loc_advantage', ['rel_size', 'loc_advantage'])

# 2) Alternative model using ratio instead of difference
m2 = run_logit('win ~ size_ratio + loc_advantage', ['size_ratio', 'loc_advantage'])

# 3) Model with both size and location plus a simple control for total group size of focal
m3 = run_logit('win ~ rel_size + loc_advantage + n_focal', ['rel_size', 'loc_advantage', 'n_focal'])

# 4) As a robustness check, standardize predictors and re-fit model 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Z = scaler.fit_transform(df[['rel_size', 'loc_advantage']])
Z_df = pd.DataFrame(Z, columns=['rel_size_z', 'loc_advantage_z'])
Xz = sm.add_constant(Z_df)

print('\n' + '=' * 80)
print('Model: win ~ standardized rel_size + standardized loc_advantage')
model_z = sm.Logit(df['win'], Xz)
res_z = model_z.fit(disp=False)
print(res_z.summary())
print('\nStandardized coefficients and p-values:')
for col in res_z.params.index:
    print(f"  {col:25s} coef={res_z.params[col]:.3f}, p={res_z.pvalues[col]:.3f}")
