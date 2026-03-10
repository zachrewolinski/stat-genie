import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'soccer.csv'

df = pd.read_csv(DATA_PATH)

# compute mean skin tone
skin = df[['rater1','rater2']].mean(axis=1)

df = df.assign(skin=skin)

# Aggregate to player level to avoid repeated dyads per referee
agg_cols = {
    'games':'sum',
    'redCards':'sum',
    'yellowCards':'sum',
    'yellowReds':'sum',
    'skin':'mean',
    'height':'mean',
    'weight':'mean',
}
player_df = df.groupby('playerShort', as_index=False).agg(agg_cols)

# basic counts
player_df = player_df.dropna(subset=['skin'])

# binary dark vs light using midpoint 0.5
player_df['dark'] = (player_df['skin'] > 0.5).astype(int)

# add slight exposure for zero games? (shouldn't be zero)
player_df = player_df[player_df['games'] > 0]

# rate per game
player_df['red_rate'] = player_df['redCards'] / player_df['games']

# summary stats
summary = player_df.groupby('dark').agg(
    n_players=('playerShort','nunique'),
    games=('games','sum'),
    redCards=('redCards','sum'),
    mean_rate=('red_rate','mean')
)

# Poisson regression with offset log(games)
player_df['log_games'] = np.log(player_df['games'])

model = smf.glm('redCards ~ dark', data=player_df, 
                family=sm.families.Poisson(), offset=player_df['log_games']).fit()

# Negative binomial as robustness
nb_model = smf.glm('redCards ~ dark', data=player_df, 
                   family=sm.families.NegativeBinomial(alpha=1.0), offset=player_df['log_games']).fit()

# Also continuous skin tone
model_cont = smf.glm('redCards ~ skin', data=player_df, 
                     family=sm.families.Poisson(), offset=player_df['log_games']).fit()

# Export results to a small text file
with open('analysis_results.txt','w') as f:
    f.write('Summary by dark (0=light/medium,1=dark):\n')
    f.write(summary.to_string())
    f.write('\n\nPoisson dark coef:')
    f.write(str(model.params.to_dict()))
    f.write('\nPoisson dark pvalues:')
    f.write(str(model.pvalues.to_dict()))
    f.write('\nPoisson dark CI:')
    f.write(str(model.conf_int().to_dict()))
    f.write('\n\nNB dark coef:')
    f.write(str(nb_model.params.to_dict()))
    f.write('\nNB dark pvalues:')
    f.write(str(nb_model.pvalues.to_dict()))
    f.write('\nNB dark CI:')
    f.write(str(nb_model.conf_int().to_dict()))
    f.write('\n\nPoisson skin coef:')
    f.write(str(model_cont.params.to_dict()))
    f.write('\nPoisson skin pvalues:')
    f.write(str(model_cont.pvalues.to_dict()))
    f.write('\nPoisson skin CI:')
    f.write(str(model_cont.conf_int().to_dict()))

print(summary)
print(model.summary())
print(nb_model.summary())
print(model_cont.summary())
