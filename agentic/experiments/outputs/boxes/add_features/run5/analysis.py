import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# Load data

df = pd.read_csv('boxes.csv')

# Keep rows with age for developmental analysis

df_age = df.dropna(subset=['age']).copy()

# Outcomes
# y: 1=unchosen, 2=majority, 3=minority

df_age['social'] = (df_age['y'] != 1).astype(int)
# among all: majority choice

df_age['majority'] = (df_age['y'] == 2).astype(int)

# Center age for interpretability

df_age['age_c'] = df_age['age'] - df_age['age'].mean()

# Descriptives

def rate(series):
    return series.mean()

# Overall rates

overall_social = rate(df_age['social'])
overall_majority = rate(df_age.loc[df_age['social'] == 1, 'majority'])

# By culture

culture_summary = (
    df_age.groupby('culture')
    .apply(lambda g: pd.Series({
        'n': len(g),
        'social_rate': rate(g['social']),
        'majority_rate_among_social': rate(g.loc[g['social'] == 1, 'majority'])
    }))
    .reset_index()
)

# By age bins

bins = [0, 14, 17, 25, 40, 100]
labels = ['11-14', '15-17', '18-25', '26-40', '41+']


df_age['age_bin'] = pd.cut(df_age['age'], bins=bins, labels=labels, include_lowest=True)

age_summary = (
    df_age.groupby('age_bin')
    .apply(lambda g: pd.Series({
        'n': len(g),
        'social_rate': rate(g['social']),
        'majority_rate_among_social': rate(g.loc[g['social'] == 1, 'majority'])
    }))
    .reset_index()
)

# Logistic models: social reliance

model_social_full = smf.glm('social ~ age_c * C(culture)', data=df_age, family=sm.families.Binomial()).fit()
model_social_reduced = smf.glm('social ~ age_c + C(culture)', data=df_age, family=sm.families.Binomial()).fit()
model_social_culture = smf.glm('social ~ C(culture)', data=df_age, family=sm.families.Binomial()).fit()

# LRT for age main effect and interaction

lrt_age_social = 2 * (model_social_reduced.llf - model_social_culture.llf)

df_age_social = model_social_reduced.df_model - model_social_culture.df_model
p_age_social = chi2.sf(lrt_age_social, df_age_social)

lrt_interaction_social = 2 * (model_social_full.llf - model_social_reduced.llf)

df_interaction_social = model_social_full.df_model - model_social_reduced.df_model
p_interaction_social = chi2.sf(lrt_interaction_social, df_interaction_social)

age_coef_social = model_social_reduced.params['age_c']

# Logistic models: majority preference among social choices

social_df = df_age[df_age['social'] == 1].copy()
model_maj_full = smf.glm('majority ~ age_c * C(culture)', data=social_df, family=sm.families.Binomial()).fit()
model_maj_reduced = smf.glm('majority ~ age_c + C(culture)', data=social_df, family=sm.families.Binomial()).fit()
model_maj_culture = smf.glm('majority ~ C(culture)', data=social_df, family=sm.families.Binomial()).fit()

lrt_age_maj = 2 * (model_maj_reduced.llf - model_maj_culture.llf)

df_age_maj = model_maj_reduced.df_model - model_maj_culture.df_model
p_age_maj = chi2.sf(lrt_age_maj, df_age_maj)

lrt_interaction_maj = 2 * (model_maj_full.llf - model_maj_reduced.llf)

df_interaction_maj = model_maj_full.df_model - model_maj_reduced.df_model
p_interaction_maj = chi2.sf(lrt_interaction_maj, df_interaction_maj)

age_coef_maj = model_maj_reduced.params['age_c']

# Output

print('Overall social reliance rate:', round(overall_social, 3))
print('Overall majority preference among social:', round(overall_majority, 3))
print('\nBy culture:')
print(culture_summary.to_string(index=False))
print('\nBy age bin:')
print(age_summary.to_string(index=False))

print('\nSocial reliance model: age main effect p-value:', p_age_social)
print('Social reliance model: age coef (log-odds per year):', age_coef_social)
print('Social reliance model: age x culture interaction p-value:', p_interaction_social)

print('\nMajority preference model: age main effect p-value:', p_age_maj)
print('Majority preference model: age coef (log-odds per year):', age_coef_maj)
print('Majority preference model: age x culture interaction p-value:', p_interaction_maj)
