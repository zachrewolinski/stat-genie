import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('boxes.csv')

# Rename columns for clarity
df = df.rename(
    columns={
        'feature1': 'outcome',
        'feature2': 'gender',
        'feature3': 'age',
        'feature4': 'majority_first',
        'feature5': 'site',
    }
)

# Derived indicators
# outcome: 1=undemonstrated option, 2=majority option, 3=minority option

# Reliance on social information: any demonstrated option (2 or 3) vs undemonstrated (1)
df['social_choice'] = (df['outcome'] != 1).astype(int)

# Preference for majority among all choices: majority (2) vs other (1 or 3)
df['majority_choice'] = (df['outcome'] == 2).astype(int)

# Preference for majority among social learners only (2 vs 3)
df_social = df[df['social_choice'] == 1].copy()
df_social['majority_pref'] = (df_social['outcome'] == 2).astype(int)


results = {}


def lr_test(full_model, reduced_model, name):
    """Likelihood-ratio test comparing nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return {
        'test': name,
        'lr_stat': float(lr_stat),
        'df_diff': int(df_diff),
        'p_value': float(p_value),
    }


# 1. Social vs asocial choice: logistic regression with age and site
model_social_full = smf.logit('social_choice ~ age + C(site)', data=df).fit(disp=False)
model_social_age_only = smf.logit('social_choice ~ age', data=df).fit(disp=False)

results['social_age_coef'] = {
    'coef': float(model_social_full.params['age']),
    'p_value': float(model_social_full.pvalues['age']),
}

results['social_site_effect'] = lr_test(
    full_model=model_social_full,
    reduced_model=model_social_age_only,
    name='Effect of site on social_choice',
)

# 2. Majority vs not (1 or 3): logistic with age and site
model_maj_full = smf.logit('majority_choice ~ age + C(site)', data=df).fit(disp=False)
model_maj_age_only = smf.logit('majority_choice ~ age', data=df).fit(disp=False)

results['majority_age_coef'] = {
    'coef': float(model_maj_full.params['age']),
    'p_value': float(model_maj_full.pvalues['age']),
}

results['majority_site_effect'] = lr_test(
    full_model=model_maj_full,
    reduced_model=model_maj_age_only,
    name='Effect of site on majority_choice',
)

# 3. Majority vs minority among social learners: logistic with age and site
model_pref_full = smf.logit('majority_pref ~ age + C(site)', data=df_social).fit(disp=False)
model_pref_age_only = smf.logit('majority_pref ~ age', data=df_social).fit(disp=False)

results['pref_age_coef'] = {
    'coef': float(model_pref_full.params['age']),
    'p_value': float(model_pref_full.pvalues['age']),
}

results['pref_site_effect'] = lr_test(
    full_model=model_pref_full,
    reduced_model=model_pref_age_only,
    name='Effect of site on majority_pref among social learners',
)

# Also examine simple descriptive variation across age quartiles and sites
age_quartiles = pd.qcut(df['age'], 4, duplicates='drop')
site_outcome = pd.crosstab(df['site'], df['outcome'], normalize='index')
age_outcome = pd.crosstab(age_quartiles, df['outcome'], normalize='index')

results['site_outcome_props'] = site_outcome.to_dict(orient='index')
results['age_outcome_props'] = {
    str(idx): row.to_dict() for idx, row in age_outcome.iterrows()
}

# Print a concise summary for manual interpretation
print('Social vs asocial choice (any demo vs undemonstrated):')
print(
    '  Age coef: {coef:.3f}, p={p_value:.4f}'.format(
        **results['social_age_coef']
    )
)
print(
    '  Site LR test: LR={lr_stat:.2f}, df={df_diff}, p={p_value:.4f}'.format(
        **results['social_site_effect']
    )
)

print('\nMajority vs other (1 or 3):')
print(
    '  Age coef: {coef:.3f}, p={p_value:.4f}'.format(
        **results['majority_age_coef']
    )
)
print(
    '  Site LR test: LR={lr_stat:.2f}, df={df_diff}, p={p_value:.4f}'.format(
        **results['majority_site_effect']
    )
)

print('\nMajority vs minority among social learners:')
print(
    '  Age coef: {coef:.3f}, p={p_value:.4f}'.format(
        **results['pref_age_coef']
    )
)
print(
    '  Site LR test: LR={lr_stat:.2f}, df={df_diff}, p={p_value:.4f}'.format(
        **results['pref_site_effect']
    )
)

print('\nOutcome proportions by site (rows sum to 1):')
print(site_outcome)

print('\nOutcome proportions by age quartile (rows sum to 1):')
print(age_outcome)
