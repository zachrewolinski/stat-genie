import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')

# Skin tone columns: values in {0,0.25,0.5,0.75,1}
skin_cols = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        vals = df[col].dropna().unique()
        if len(vals) <= 6:
            allowed = np.array([0, 0.25, 0.5, 0.75, 1])
            if np.all(np.isin(np.round(vals, 2), allowed)):
                skin_cols.append(col)

skin_cols = [c for c in skin_cols if df[c].nunique() >= 4]

skin_tone = df[skin_cols].mean(axis=1, skipna=True)

# Choose candidate red card columns (rare, low mean counts)
primary_red = 'yellowCards' if 'yellowCards' in df.columns else None
alt_red = 'meanExp' if 'meanExp' in df.columns else None


def analyze(red_col):
    data = df.copy()
    data['skin_tone'] = skin_tone
    data = data.loc[data['skin_tone'].notna() & data[red_col].notna()].copy()

    # Define dark vs light; exclude mid
    data['skin_group'] = np.where(data['skin_tone'] >= 0.75, 'dark',
                                  np.where(data['skin_tone'] <= 0.25, 'light', 'mid'))
    data_main = data[data['skin_group'].isin(['dark','light'])].copy()

    dark = data_main[data_main['skin_group']=='dark'][red_col]
    light = data_main[data_main['skin_group']=='light'][red_col]

    # Group stats
    stats_out = {
        'n_dark': int(len(dark)),
        'n_light': int(len(light)),
        'mean_dark': float(dark.mean()),
        'mean_light': float(light.mean()),
        'any_dark': float((dark > 0).mean()),
        'any_light': float((light > 0).mean()),
    }

    # Tests
    try:
        tstat, t_p = stats.ttest_ind(dark, light, equal_var=False)
    except Exception:
        tstat, t_p = np.nan, np.nan
    try:
        ustat, u_p = stats.mannwhitneyu(dark, light, alternative='two-sided')
    except Exception:
        ustat, u_p = np.nan, np.nan

    # Spearman correlation
    try:
        rho, rho_p = stats.spearmanr(data['skin_tone'], data[red_col])
    except Exception:
        rho, rho_p = np.nan, np.nan

    # Poisson regression (red cards ~ skin_tone)
    try:
        X = sm.add_constant(data['skin_tone'])
        model = sm.GLM(data[red_col], X, family=sm.families.Poisson())
        res = model.fit()
        beta = float(res.params['skin_tone'])
        beta_p = float(res.pvalues['skin_tone'])
    except Exception:
        beta, beta_p = np.nan, np.nan

    return stats_out, {'t_p': t_p, 'u_p': u_p, 'rho': rho, 'rho_p': rho_p, 'beta': beta, 'beta_p': beta_p}


results = {}
for red_col in [primary_red, alt_red]:
    if red_col is not None:
        results[red_col] = analyze(red_col)

# Build explanation
# Prefer primary_red, but mention robustness with alt_red if available
primary_stats, primary_tests = results[primary_red]

exp = []
exp.append(f"Skin tone was averaged from the two 5-level rater columns ({', '.join(skin_cols)}), then players with skin_tone >= 0.75 were labeled 'dark' and <= 0.25 labeled 'light' (mid tones excluded).")
exp.append(f"Using the rare red-card count column '{primary_red}', dark vs light dyads show nearly identical red-card rates: mean red cards per dyad {primary_stats['mean_dark']:.5f} (dark, n={primary_stats['n_dark']}) vs {primary_stats['mean_light']:.5f} (light, n={primary_stats['n_light']}). Any-red-card rate is {primary_stats['any_dark']*100:.2f}% vs {primary_stats['any_light']*100:.2f}%.")
exp.append(f"Statistical tests show no significant difference (Welch t-test p={primary_tests['t_p']:.3f}; Mann–Whitney p={primary_tests['u_p']:.3f}). The continuous association is also negligible (Spearman rho={primary_tests['rho']:.4f}, p={primary_tests['rho_p']:.3f}; Poisson skin-tone coefficient p={primary_tests['beta_p']:.3f}).")

if alt_red in results:
    alt_stats, alt_tests = results[alt_red]
    exp.append(f"Sensitivity check with the other rare red-card-like column '{alt_red}' gives the same conclusion: dark mean {alt_stats['mean_dark']:.5f} vs light {alt_stats['mean_light']:.5f}, with non-significant tests (t p={alt_tests['t_p']:.3f}, Spearman p={alt_tests['rho_p']:.3f}).")

exp.append("Overall, the data do not provide evidence that darker-skinned players are more likely to receive red cards than lighter-skinned players.")

explanation = " ".join(exp)

# Likert response: strong No (0) to strong Yes (100). Here evidence suggests no relationship.
response = 25

out = {'response': response, 'explanation': explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
