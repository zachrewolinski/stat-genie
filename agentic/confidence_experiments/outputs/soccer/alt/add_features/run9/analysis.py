import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = 'soccer.csv'

# Load data
_df = pd.read_csv(DATA_PATH)

# Skin tone: average of raters (use available rater values)
skin = _df[['rater1', 'rater2']].mean(axis=1, skipna=True)
_df = _df.assign(skin=skin)

# Keep rows with skin ratings and valid games
_df = _df[_df['skin'].notna()].copy()
_df = _df[_df['games'] > 0].copy()

# Define dark vs light using 0.5 threshold (neutral excluded)
_df['dark'] = _df['skin'] > 0.5
_df['light'] = _df['skin'] < 0.5

# Main analysis dataset: exclude neutral (skin == 0.5)
main = _df[_df['skin'] != 0.5].copy()

# Outcome counts and exposure
main['redCards'] = main['redCards'].fillna(0)
main['games'] = main['games'].fillna(0)

# Summary rates
summary = main.groupby('dark').agg(
    dyads=('redCards', 'size'),
    total_red=('redCards', 'sum'),
    total_games=('games', 'sum'),
    red_any=('redCards', lambda x: (x > 0).sum())
)
summary['rate_per_game'] = summary['total_red'] / summary['total_games']
summary['rate_per_100'] = summary['rate_per_game'] * 100

# Poisson regression with exposure (games) as offset
# Use robust SE to be conservative
poisson_model = smf.glm(
    formula='redCards ~ dark',
    data=main,
    family=sm.families.Poisson(),
    offset=np.log(main['games'])
).fit(cov_type='HC0')

coef = poisson_model.params['dark[T.True]'] if 'dark[T.True]' in poisson_model.params else poisson_model.params.get('dark', np.nan)
se = poisson_model.bse['dark[T.True]'] if 'dark[T.True]' in poisson_model.bse else poisson_model.bse.get('dark', np.nan)
rr = float(np.exp(coef))
ci_low, ci_high = np.exp(coef + np.array([-1, 1]) * 1.96 * se)

# Two-sample rate ratio test using Poisson counts (approx)
# Use summary stats for an additional check
# (Not used for decision, but reported in explanation if needed)

# Logistic regression for any red card (robustness)
main['red_any'] = (main['redCards'] > 0).astype(int)
logit_model = smf.logit('red_any ~ dark + np.log(games)', data=main).fit(disp=0)
logit_coef = logit_model.params['dark[T.True]'] if 'dark[T.True]' in logit_model.params else logit_model.params.get('dark', np.nan)
logit_se = logit_model.bse['dark[T.True]'] if 'dark[T.True]' in logit_model.bse else logit_model.bse.get('dark', np.nan)
logit_or = float(np.exp(logit_coef))
logit_ci_low, logit_ci_high = np.exp(logit_coef + np.array([-1, 1]) * 1.96 * logit_se)

# Sensitivity: include neutral (skin==0.5) in light group
sens = _df.copy()
sens['dark'] = sens['skin'] > 0.5
sens['redCards'] = sens['redCards'].fillna(0)
poisson_sens = smf.glm(
    formula='redCards ~ dark',
    data=sens,
    family=sm.families.Poisson(),
    offset=np.log(sens['games'])
).fit(cov_type='HC0')
coef_sens = poisson_sens.params['dark[T.True]'] if 'dark[T.True]' in poisson_sens.params else poisson_sens.params.get('dark', np.nan)
se_sens = poisson_sens.bse['dark[T.True]'] if 'dark[T.True]' in poisson_sens.bse else poisson_sens.bse.get('dark', np.nan)
rr_sens = float(np.exp(coef_sens))
ci_sens_low, ci_sens_high = np.exp(coef_sens + np.array([-1, 1]) * 1.96 * se_sens)

# Build results dictionary
results = {
    'n_total': int(len(_df)),
    'n_main': int(len(main)),
    'summary_by_dark': summary.reset_index().to_dict(orient='records'),
    'poisson_rr': rr,
    'poisson_ci_low': float(ci_low),
    'poisson_ci_high': float(ci_high),
    'poisson_pvalue': float(poisson_model.pvalues.get('dark[T.True]', poisson_model.pvalues.get('dark', np.nan))),
    'logit_or': logit_or,
    'logit_ci_low': float(logit_ci_low),
    'logit_ci_high': float(logit_ci_high),
    'logit_pvalue': float(logit_model.pvalues.get('dark[T.True]', logit_model.pvalues.get('dark', np.nan))),
    'sens_poisson_rr': rr_sens,
    'sens_poisson_ci_low': float(ci_sens_low),
    'sens_poisson_ci_high': float(ci_sens_high),
    'sens_poisson_pvalue': float(poisson_sens.pvalues.get('dark[T.True]', poisson_sens.pvalues.get('dark', np.nan))),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
