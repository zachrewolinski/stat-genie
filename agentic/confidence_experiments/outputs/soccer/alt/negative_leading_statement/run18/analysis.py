import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv('soccer.csv')

    # Skin tone average (0-1), using available raters
    df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)
    # Keep rows with skin tone info and games > 0
    df = df[(~df['skin_tone'].isna()) & (df['games'] > 0)]

    # Define light/dark categories for descriptive comparison
    # Light: <=0.25, Dark: >=0.75 (extremes on 5-point scale)
    df['skin_cat'] = np.where(df['skin_tone'] <= 0.25, 'light',
                        np.where(df['skin_tone'] >= 0.75, 'dark', 'mid'))

    # Descriptive rate per game by category
    rates = (
        df.groupby('skin_cat')
          .apply(lambda g: pd.Series({
              'dyads': len(g),
              'games': g['games'].sum(),
              'red_cards': g['redCards'].sum(),
              'rate_per_game': g['redCards'].sum() / g['games'].sum() if g['games'].sum() > 0 else np.nan,
          }))
          .reset_index()
    )

    # Poisson regression on dyad-level counts with log(games) offset
    df['log_games'] = np.log(df['games'])

    # Model 1: skin_tone only
    model1 = smf.glm(
        'redCards ~ skin_tone',
        data=df,
        family=sm.families.Poisson(),
        offset=df['log_games']
    ).fit(cov_type='HC1')

    # Model 2: add controls for position and leagueCountry
    model2 = smf.glm(
        'redCards ~ skin_tone + C(position) + C(leagueCountry)',
        data=df,
        family=sm.families.Poisson(),
        offset=df['log_games']
    ).fit(cov_type='HC1')

    # Create light vs dark comparison using model2 predictions
    # Build representative rows for light (0.25) and dark (0.75) using most common categories
    modal_position = df['position'].mode().iloc[0]
    modal_league = df['leagueCountry'].mode().iloc[0]
    base = pd.DataFrame({
        'skin_tone': [0.25, 0.75],
        'position': [modal_position, modal_position],
        'leagueCountry': [modal_league, modal_league],
        'games': [1, 1],
    })
    base['log_games'] = 0.0
    pred = model2.get_prediction(base).summary_frame()

    # Print outputs for later use
    print('DESCRIPTIVE_RATES')
    print(rates.to_string(index=False))
    print('\nMODEL1_COEF')
    print(model1.params.to_string())
    print('\nMODEL1_PVALUES')
    print(model1.pvalues.to_string())
    print('\nMODEL2_COEF')
    print(model2.params.to_string())
    print('\nMODEL2_PVALUES')
    print(model2.pvalues.to_string())
    print('\nMODEL2_PRED_LIGHT_DARK')
    print(pd.concat([base[['skin_tone']], pred[['mean', 'mean_ci_lower', 'mean_ci_upper']]], axis=1).to_string(index=False))

if __name__ == '__main__':
    main()
