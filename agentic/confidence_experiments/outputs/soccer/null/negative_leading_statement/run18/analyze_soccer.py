import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'soccer.csv'

df = pd.read_csv(path)
for col in ['rater1','rater2','games','redCards']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['skinTone'] = df[['rater1','rater2']].mean(axis=1)
analysis = df.dropna(subset=['skinTone','games','redCards']).copy()
analysis = analysis[analysis['games'] > 0]

analysis['skinGroup'] = np.where(analysis['skinTone'] > 0.5, 'dark', 'light_or_medium')
analysis['skinGroup_strict'] = np.select(
    [analysis['skinTone'] <= 0.25, analysis['skinTone'] >= 0.75],
    ['light', 'dark'],
    default='mid'
)

summary = analysis.groupby('skinGroup').agg(
    dyads=('skinGroup','size'),
    total_games=('games','sum'),
    total_red=('redCards','sum')
)
summary['red_per_game'] = summary['total_red'] / summary['total_games']
summary['red_per_100_games'] = summary['red_per_game'] * 100

summary_strict = analysis[analysis['skinGroup_strict']!='mid'].groupby('skinGroup_strict').agg(
    dyads=('skinGroup_strict','size'),
    total_games=('games','sum'),
    total_red=('redCards','sum')
)
summary_strict['red_per_game'] = summary_strict['total_red'] / summary_strict['total_games']
summary_strict['red_per_100_games'] = summary_strict['red_per_game'] * 100

analysis['log_games'] = np.log(analysis['games'])

model_cont = smf.glm('redCards ~ skinTone', data=analysis, family=sm.families.Poisson(), offset=analysis['log_games']).fit(cov_type='HC0')

analysis['dark'] = (analysis['skinTone'] > 0.5).astype(int)
model_bin = smf.glm('redCards ~ dark', data=analysis, family=sm.families.Poisson(), offset=analysis['log_games']).fit(cov_type='HC0')

analysis_strict = analysis[analysis['skinGroup_strict']!='mid'].copy()
analysis_strict['dark'] = (analysis_strict['skinGroup_strict']=='dark').astype(int)
model_strict = smf.glm('redCards ~ dark', data=analysis_strict, family=sm.families.Poisson(), offset=np.log(analysis_strict['games'])).fit(cov_type='HC0')


def rate_ratio(model, param):
    coef = model.params[param]
    se = model.bse[param]
    rr = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    p = model.pvalues[param]
    return rr, ci_low, ci_high, p

rr_cont, lo_cont, hi_cont, p_cont = rate_ratio(model_cont, 'skinTone')
rr_bin, lo_bin, hi_bin, p_bin = rate_ratio(model_bin, 'dark')
rr_strict, lo_strict, hi_strict, p_strict = rate_ratio(model_strict, 'dark')

print('N rows:', len(analysis))
print('Summary (threshold >0.5 dark):')
print(summary)
print('\nSummary strict (<=0.25 light vs >=0.75 dark):')
print(summary_strict)

print('\nPoisson continuous skinTone (per 1.0 increase in skinTone):')
print('RR', rr_cont, 'CI', (lo_cont, hi_cont), 'p', p_cont)

print('\nPoisson binary dark (>0.5 vs <=0.5):')
print('RR', rr_bin, 'CI', (lo_bin, hi_bin), 'p', p_bin)

print('\nPoisson strict dark (>=0.75) vs light (<=0.25):')
print('RR', rr_strict, 'CI', (lo_strict, hi_strict), 'p', p_strict)
