import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'crofoot.csv'
df = pd.read_csv(path)

# relative variables

df['rel_size'] = df['n_focal'] - df['n_other']
df['rel_loc'] = df['dist_other'] - df['dist_focal']  # positive => closer to focal


# helper to fit logit

def fit_logit(cols):
    X = sm.add_constant(df[cols])
    y = df['win']
    model = sm.Logit(y, X).fit(disp=False)
    return model

models = {
    'rel_size_only': ['rel_size'],
    'rel_loc_only': ['rel_loc'],
    'rel_size_rel_loc': ['rel_size','rel_loc'],
    'full_size_loc': ['n_focal','n_other','dist_focal','dist_other']
}

for name, cols in models.items():
    model = fit_logit(cols)
    print('\n', name)
    print(model.summary())

    params = model.params
    conf = model.conf_int()
    odds = np.exp(params)
    conf_odds = np.exp(conf)
    print('\nOdds ratios:')
    print(odds)
    print('95% CI:')
    print(conf_odds)

