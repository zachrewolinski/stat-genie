import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='crofoot.csv'
df=pd.read_csv(path)
cols=['win','n_focal','n_other','dist_focal','dist_other']
df=df.dropna(subset=cols)
df['rel_size']=df['n_focal']-df['n_other']
df['loc_adv']=df['dist_other']-df['dist_focal']
for c in ['rel_size','loc_adv']:
    df[c+'_z']=(df[c]-df[c].mean())/df[c].std(ddof=0)
model = smf.glm('win ~ rel_size_z + loc_adv_z', data=df, family=sm.families.Binomial()).fit()
print('llf', model.llf)
print('llnull', model.llnull)
print('deviance', model.deviance, 'null_deviance', model.null_deviance)
print('aic', model.aic)
