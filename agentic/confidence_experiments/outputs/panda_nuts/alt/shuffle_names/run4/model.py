import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns to meanings based on metadata + observed values
# age: age in years
# nuts_opened count: column 'help'
# seconds duration: column 'chimpanzee'
# sex: column 'nuts_opened' (m/f)
# help received: column 'seconds' (y/N)
# chimpanzee ID: column 'hammer' (numeric id)

mapped = df.rename(columns={
    'help': 'nuts_opened_count',
    'chimpanzee': 'duration_seconds',
    'nuts_opened': 'sex_mf',
    'seconds': 'help_received',
    'hammer': 'chimp_id',
    'sex': 'hammer_type'
})

# Filter out zero or negative durations if any
mapped = mapped[mapped['duration_seconds'] > 0].copy()

# Efficiency: nuts per second
mapped['efficiency'] = mapped['nuts_opened_count'] / mapped['duration_seconds']

# Clean categorical variables
mapped['sex_mf'] = mapped['sex_mf'].astype('category')
mapped['help_received'] = mapped['help_received'].astype('category')

# Basic summaries
summary = {
    'n': len(mapped),
    'eff_mean': mapped['efficiency'].mean(),
    'eff_median': mapped['efficiency'].median(),
    'eff_std': mapped['efficiency'].std(),
}
print('summary', summary)

# OLS with cluster-robust SE by chimp_id (repeated measures)
model = smf.ols('efficiency ~ age + C(sex_mf) + C(help_received)', data=mapped).fit(
    cov_type='cluster',
    cov_kwds={'groups': mapped['chimp_id']}
)

print(model.summary())

# Also check model without cluster for reference
model_hc3 = smf.ols('efficiency ~ age + C(sex_mf) + C(help_received)', data=mapped).fit(cov_type='HC3')
print('\nHC3 summary')
print(model_hc3.summary())

# Compute group means for context
means = mapped.groupby(['sex_mf', 'help_received'])['efficiency'].mean()
print('\nGroup means (efficiency by sex/help)')
print(means)

# Age correlation
print('\nAge correlation with efficiency')
print(mapped[['age', 'efficiency']].corr().iloc[0,1])

