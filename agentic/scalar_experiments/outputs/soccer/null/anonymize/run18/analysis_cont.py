import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('soccer.csv')
df['skin_tone'] = df[['feature18','feature19']].mean(axis=1, skipna=True)
player = (
    df.groupby('feature1', as_index=False)
    .agg(skin_tone=('skin_tone','mean'), red_cards=('feature16','sum'), games=('feature9','sum'))
)
player = player[~player['skin_tone'].isna() & (player['games']>0)]

# Continuous model
player['log_games'] = np.log(player['games'])
model = sm.GLM(player['red_cards'], sm.add_constant(player['skin_tone']), family=sm.families.Poisson(), offset=player['log_games'])
res = model.fit()
coef = res.params['skin_tone']
rr = float(np.exp(coef))
ci = res.conf_int().loc['skin_tone']
rr_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
pval = float(res.pvalues['skin_tone'])

# Quartile comparison
q1 = player['skin_tone'].quantile(0.25)
q3 = player['skin_tone'].quantile(0.75)
player['skin_group'] = np.where(player['skin_tone']<=q1,'light', np.where(player['skin_tone']>=q3,'dark','mid'))
ld = player[player['skin_group'].isin(['light','dark'])].copy()
ld['dark'] = (ld['skin_group']=='dark').astype(int)
ld['log_games'] = np.log(ld['games'])
model2 = sm.GLM(ld['red_cards'], sm.add_constant(ld['dark']), family=sm.families.Poisson(), offset=ld['log_games'])
res2 = model2.fit()
coef2 = res2.params['dark']
rr2 = float(np.exp(coef2))
ci2 = res2.conf_int().loc['dark']
rr2_ci = (float(np.exp(ci2[0])), float(np.exp(ci2[1])))
pval2 = float(res2.pvalues['dark'])

# Summary rates
summary = (
    ld.groupby('skin_group')
    .apply(lambda x: pd.Series({
        'players': len(x),
        'total_red_cards': x['red_cards'].sum(),
        'total_games': x['games'].sum(),
        'red_cards_per_game': x['red_cards'].sum()/x['games'].sum()
    }))
    .reset_index()
)

print({
    'n_players': len(player),
    'skin_tone_range': (float(player['skin_tone'].min()), float(player['skin_tone'].max())),
    'coef': float(coef),
    'rr_per_1_unit': rr,
    'rr_ci': rr_ci,
    'pval': pval,
    'q1': float(q1),
    'q3': float(q3),
    'rr_q': rr2,
    'rr_q_ci': rr2_ci,
    'pval_q': pval2,
    'summary_q': summary,
})
