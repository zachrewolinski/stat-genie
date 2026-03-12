import pandas as pd
import statsmodels.formula.api as smf

# prepare
raw = pd.read_csv('amtl.csv')
df = raw.rename(columns={'genus':'amtl','pop':'age_at_death','stdev_age':'prob_male','sockets':'tooth_class','tooth_class':'genus'})
df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens','Pan','Papio','Pongo'])
df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior','Posterior','Premolar'])

formulas = [
    'amtl ~ C(genus)',
    'amtl ~ C(tooth_class)',
    'amtl ~ age_at_death + prob_male',
    'amtl ~ C(genus) + age_at_death + prob_male',
    'amtl ~ C(genus) + C(tooth_class)',
    'amtl ~ C(genus) + age_at_death + prob_male + C(tooth_class)'
]

for f in formulas:
    try:
        smf.ols(f, data=df).fit()
        print('OK', f)
    except Exception as e:
        print('FAIL', f, type(e), e)

