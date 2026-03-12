import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('crofoot.csv')

# Map columns using info.json descriptions (see AGENTS.md instructions)
# Outcome: m_focal = 1 if focal won contest
# Focal group size: f_other
# Other group size: win
# Distance of focal group from its home range center: m_other
# Distance of other group from its home range center: n_focal

_df['relative_size'] = _df['f_other'] - _df['win']
_df['relative_location'] = _df['n_focal'] - _df['m_other']

# Logistic regression: win ~ relative_size + relative_location
X = _df[['relative_size', 'relative_location']]
X = sm.add_constant(X)
y = _df['m_focal']

model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())

# Odds ratios and confidence intervals
params = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']

odds = params.apply(lambda v: float('nan') if pd.isna(v) else float(pd.np.exp(v)))
conf_odds = conf.applymap(lambda v: float(pd.np.exp(v)))

print('\nOdds ratios:')
print(odds)
print('\nOdds ratio 95% CI:')
print(conf_odds)

# Also check univariate models for robustness
for var in ['relative_size', 'relative_location']:
    X1 = sm.add_constant(_df[[var]])
    res1 = sm.Logit(y, X1).fit(disp=False)
    print(f"\nUnivariate model for {var}:")
    print(res1.summary())
