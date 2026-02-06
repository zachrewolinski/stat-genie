import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = 'crofoot.csv'
df = pd.read_csv(csv_path)

# Construct predictors
# Relative group size: focal group size - other group size
# Based on metadata: f_other = focal group individuals, win = other group individuals
if 'f_other' not in df.columns or 'win' not in df.columns:
    raise ValueError('Expected columns f_other and win for group sizes not found.')

# Relative location advantage: other distance from its home-range center minus focal distance
# Based on metadata: m_other = focal distance to its center, n_focal = other distance to its center
if 'm_other' not in df.columns or 'n_focal' not in df.columns:
    raise ValueError('Expected columns m_other and n_focal for distances not found.')


df['rel_size'] = df['f_other'] - df['win']
df['rel_location'] = df['n_focal'] - df['m_other']

# Outcome: focal win
if 'm_focal' not in df.columns:
    raise ValueError('Expected column m_focal for win outcome not found.')

y = df['m_focal']
X = df[['rel_size', 'rel_location']]
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(disp=False)

# Output key results
summary = result.summary2().tables[1]
print('Logistic regression: m_focal ~ rel_size + rel_location')
print(summary)

# Also compute simple descriptive stats for win rate by sign of predictors
for col in ['rel_size', 'rel_location']:
    df['sign'] = df[col].apply(lambda v: 'positive' if v > 0 else ('negative' if v < 0 else 'zero'))
    win_rates = df.groupby('sign')['m_focal'].mean()
    print(f"\nWin rate by {col} sign:")
    print(win_rates)
