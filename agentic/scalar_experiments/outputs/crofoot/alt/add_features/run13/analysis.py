import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('crofoot.csv')
print('columns', df.columns.tolist())
print('shape', df.shape)
print(df.head())

# derive relative size and location
if {'n_focal','n_other','dist_focal','dist_other','win'}.issubset(df.columns):
    df = df.copy()
    df['rel_size'] = df['n_focal'] - df['n_other']
    df['rel_size_ratio'] = df['n_focal'] / df['n_other']
    df['rel_location'] = df['dist_other'] - df['dist_focal']  # positive => focal closer to its center
    df['focal_closer'] = (df['dist_focal'] < df['dist_other']).astype(int)

    print(df[['win','n_focal','n_other','dist_focal','dist_other','rel_size','rel_location','focal_closer']].describe())

    # logistic regression with continuous predictors
    model1 = smf.logit('win ~ rel_size + rel_location', data=df).fit(disp=False)
    print('\nModel1 summary')
    print(model1.summary())

    # logistic regression with ratio instead of difference
    model2 = smf.logit('win ~ rel_size_ratio + rel_location', data=df).fit(disp=False)
    print('\nModel2 summary')
    print(model2.summary())

    # logistic regression with binary location
    model3 = smf.logit('win ~ rel_size + focal_closer', data=df).fit(disp=False)
    print('\nModel3 summary')
    print(model3.summary())

    # compute simple effects: win rate by focal_closer and by rel_size sign
    win_rate_closer = df.groupby('focal_closer')['win'].mean()
    print('\nwin rate by focal_closer')
    print(win_rate_closer)

    df['rel_size_positive'] = (df['rel_size'] > 0).astype(int)
    print('\nwin rate by rel_size_positive')
    print(df.groupby('rel_size_positive')['win'].mean())

    # odds ratios
    print('\nOdds ratios model1')
    print(np.exp(model1.params))

else:
    print('Required columns missing')
