import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average from rater1 and nExp (rater2)
df['skin_tone'] = df[['rater1','nExp']].mean(axis=1)

# Use only rows with skin rating and positive games
analysis_df = df.dropna(subset=['skin_tone']).copy()
analysis_df = analysis_df[analysis_df['redCards'] > 0].copy()

# Red cards per game (yellowCards per description)
analysis_df['red_per_game'] = analysis_df['yellowCards'] / analysis_df['redCards']

# Median split for light vs dark
median_skin = analysis_df['skin_tone'].median()
analysis_df['skin_group'] = np.where(analysis_df['skin_tone'] > median_skin, 'dark', 'light')

rates = analysis_df.groupby('skin_group')['red_per_game'].mean()
counts = analysis_df['skin_group'].value_counts()

# t-test
light = analysis_df.loc[analysis_df['skin_group']=='light', 'red_per_game']
dark = analysis_df.loc[analysis_df['skin_group']=='dark', 'red_per_game']
stat, pval = stats.ttest_ind(dark, light, equal_var=False, nan_policy='omit')

# Poisson regression with offset
analysis_df['log_games'] = np.log(analysis_df['redCards'])
poisson_model = smf.glm(
    formula='yellowCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit()

coef = poisson_model.params['skin_tone']
se = poisson_model.bse['skin_tone']
p = poisson_model.pvalues['skin_tone']
rate_ratio = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

# Proportion of dyads with >=1 red card by group
analysis_df['any_red'] = (analysis_df['yellowCards'] > 0).astype(int)
prop_any = analysis_df.groupby('skin_group')['any_red'].mean()

print("n_rows", len(analysis_df))
print("median_skin", median_skin)
print("group_counts", counts.to_dict())
print("mean_red_per_game", rates.to_dict())
print("t_stat", stat, "p", pval)
print("poisson_coef", coef, "se", se, "p", p, "rate_ratio", rate_ratio, "ci", (ci_low, ci_high))
print("prop_any_red", prop_any.to_dict())
