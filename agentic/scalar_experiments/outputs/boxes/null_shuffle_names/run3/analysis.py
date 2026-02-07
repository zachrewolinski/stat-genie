import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('boxes.csv')

df['social_choice'] = df['majority_first'].isin([2, 3]).astype(int)
df['majority_choice'] = (df['majority_first'] == 2).astype(int)

df['site'] = df['y'].astype('category')

results = {}

for outcome in ['social_choice', 'majority_choice']:
    m_age = smf.logit(f"{outcome} ~ age", data=df).fit(disp=0)
    m_site = smf.logit(f"{outcome} ~ age + C(site)", data=df).fit(disp=0)
    m_int = smf.logit(f"{outcome} ~ age * C(site)", data=df).fit(disp=0)

    lr_site = 2 * (m_site.llf - m_age.llf)
    df_site = m_site.df_model - m_age.df_model
    p_site = 1 - stats.chi2.cdf(lr_site, df_site) if df_site > 0 else np.nan

    lr_int = 2 * (m_int.llf - m_site.llf)
    df_int = m_int.df_model - m_site.df_model
    p_int = 1 - stats.chi2.cdf(lr_int, df_int) if df_int > 0 else np.nan

    r2 = 1 - (m_site.llf / m_site.llnull)

    results[outcome] = {
        'age_coef': m_site.params.get('age', np.nan),
        'age_p': m_site.pvalues.get('age', np.nan),
        'p_site': p_site,
        'p_int': p_int,
        'r2': r2,
        'n': len(df),
    }

print(results)

score = 0

soc = results['social_choice']
if soc['age_p'] < 0.05:
    score += 20
if soc['p_site'] < 0.05:
    score += 20
if soc['p_int'] < 0.05:
    score += 15
score += min(15, max(0, soc['r2'] * 100))

maj = results['majority_choice']
if maj['age_p'] < 0.05:
    score += 20
if maj['p_site'] < 0.05:
    score += 20
if maj['p_int'] < 0.05:
    score += 15
score += min(15, max(0, maj['r2'] * 100))

score = int(round(max(-100, min(100, score))))
print('score', score)
