import pandas as pd
import statsmodels.api as sm

# load

df = pd.read_csv('crofoot.csv')

# outcome
# predictors: f_other (focal size), win (other size), m_other (focal distance), n_focal (other distance)

X = df[['f_other','win','m_other','n_focal']].copy()
X = sm.add_constant(X)

y = df['m_focal']

model = sm.Logit(y, X)
res = model.fit(disp=False)

print(res.summary2().tables[1])

