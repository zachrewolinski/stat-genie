import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = "soccer.csv"
df = pd.read_csv(path)

# Identify candidate columns for red cards by examining ranges
numeric_cols = df.select_dtypes(include=["number"]).columns
summary = df[numeric_cols].agg(['min','max','mean','std','sum'])
print("Numeric summary:\n", summary)

# Skin tone from rater1 and nExp (rater2)
df['skin_tone'] = df[['rater1','nExp']].mean(axis=1)

# Determine likely games column: largest max, and integer counts
# Here, use column named 'redCards' per metadata description (games in dyad)
# but verify by max and distribution

# We'll compute red card counts for each numeric column with small max
small_max_cols = [c for c in numeric_cols if df[c].max() <= 5]
print("Small max cols (<=5):", small_max_cols)

# Show value counts for these columns
for c in small_max_cols:
    print("\n", c, df[c].describe())
    print(df[c].value_counts().head())

# For analysis, use column 'yellowCards' as red cards based on metadata description
# Compute red cards rate per game using 'redCards' column as games count

# Ensure games > 0
analysis_df = df.copy()
analysis_df = analysis_df[analysis_df['redCards'] > 0]
analysis_df['red_per_game'] = analysis_df['yellowCards'] / analysis_df['redCards']

# Split skin tone into light vs dark using median
analysis_df = analysis_df.dropna(subset=['skin_tone'])
median_skin = analysis_df['skin_tone'].median()
analysis_df['skin_group'] = np.where(analysis_df['skin_tone'] > median_skin, 'dark', 'light')

# Compare red card rates between groups
rates = analysis_df.groupby('skin_group')['red_per_game'].mean()
print("\nMean red cards per game by group:", rates)

# t-test for difference in rates
light = analysis_df.loc[analysis_df['skin_group']=='light', 'red_per_game']
dark = analysis_df.loc[analysis_df['skin_group']=='dark', 'red_per_game']

# Use Welch t-test
stat, pval = stats.ttest_ind(dark, light, equal_var=False, nan_policy='omit')
print("Welch t-test t=%.4f p=%.4g" % (stat, pval))

# Poisson regression: red cards count with offset log(games)
# Use skin_tone continuous
analysis_df = analysis_df.copy()
analysis_df['log_games'] = np.log(analysis_df['redCards'])

poisson_model = smf.glm(
    formula='yellowCards ~ skin_tone',
    data=analysis_df,
    family=sm.families.Poisson(),
    offset=analysis_df['log_games']
).fit()
print(poisson_model.summary())

# Extract effect size
coef = poisson_model.params['skin_tone']
se = poisson_model.bse['skin_tone']
p = poisson_model.pvalues['skin_tone']
rate_ratio = np.exp(coef)
print("Skin tone coef", coef, "SE", se, "p", p, "rate_ratio", rate_ratio)
