import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from scipy.stats import chi2

# Load data

df = pd.read_csv('boxes.csv')

# Create indicators
# Social information reliance: choosing either majority (2) or minority (3) vs unchosen option (1)
df['social'] = (df['y'] != 1).astype(int)
# Majority preference among social choices
social_df = df[df['social'] == 1].copy()
social_df['majority_choice'] = (social_df['y'] == 2).astype(int)
# Majority preference overall (majority vs non-majority, including unchosen)
df['majority_all'] = (df['y'] == 2).astype(int)

# Model 1: Social reliance ~ age + culture
m1 = smf.glm('social ~ age + C(culture)', data=df, family=Binomial()).fit()
# Model 1b: add age*culture interaction
m1_int = smf.glm('social ~ age * C(culture)', data=df, family=Binomial()).fit()

# Model 2: Majority preference ~ age + culture (only among social choices)
m2 = smf.glm('majority_choice ~ age + C(culture)', data=social_df, family=Binomial()).fit()
# Model 2b: add age*culture interaction
m2_int = smf.glm('majority_choice ~ age * C(culture)', data=social_df, family=Binomial()).fit()

# Likelihood ratio tests for interaction terms
lr_stat_m1 = 2 * (m1_int.llf - m1.llf)
lr_df_m1 = m1_int.df_model - m1.df_model
lr_p_m1 = chi2.sf(lr_stat_m1, lr_df_m1)

lr_stat_m2 = 2 * (m2_int.llf - m2.llf)
lr_df_m2 = m2_int.df_model - m2.df_model
lr_p_m2 = chi2.sf(lr_stat_m2, lr_df_m2)

# Extract key p-values for main effects
age_p_m1 = m1.pvalues.get('age')
# For culture, test using likelihood ratio between model with and without culture
m1_age_only = smf.glm('social ~ age', data=df, family=Binomial()).fit()
lr_stat_cult_m1 = 2 * (m1.llf - m1_age_only.llf)
lr_df_cult_m1 = m1.df_model - m1_age_only.df_model
lr_p_cult_m1 = chi2.sf(lr_stat_cult_m1, lr_df_cult_m1)

age_p_m2 = m2.pvalues.get('age')
m2_age_only = smf.glm('majority_choice ~ age', data=social_df, family=Binomial()).fit()
lr_stat_cult_m2 = 2 * (m2.llf - m2_age_only.llf)
lr_df_cult_m2 = m2.df_model - m2_age_only.df_model
lr_p_cult_m2 = chi2.sf(lr_stat_cult_m2, lr_df_cult_m2)

# Model 3: Majority preference overall ~ age + culture
m3 = smf.glm('majority_all ~ age + C(culture)', data=df, family=Binomial()).fit()
m3_int = smf.glm('majority_all ~ age * C(culture)', data=df, family=Binomial()).fit()
age_p_m3 = m3.pvalues.get('age')
m3_age_only = smf.glm('majority_all ~ age', data=df, family=Binomial()).fit()
lr_stat_cult_m3 = 2 * (m3.llf - m3_age_only.llf)
lr_df_cult_m3 = m3.df_model - m3_age_only.df_model
lr_p_cult_m3 = chi2.sf(lr_stat_cult_m3, lr_df_cult_m3)
lr_stat_m3 = 2 * (m3_int.llf - m3.llf)
lr_df_m3 = m3_int.df_model - m3.df_model
lr_p_m3 = chi2.sf(lr_stat_m3, lr_df_m3)

results = {
    'social_age_p': age_p_m1,
    'social_culture_lr_p': lr_p_cult_m1,
    'social_age_culture_lr_p': lr_p_m1,
    'majority_age_p': age_p_m2,
    'majority_culture_lr_p': lr_p_cult_m2,
    'majority_age_culture_lr_p': lr_p_m2,
    'majority_all_age_p': age_p_m3,
    'majority_all_culture_lr_p': lr_p_cult_m3,
    'majority_all_age_culture_lr_p': lr_p_m3,
    'n_total': len(df),
    'n_social': len(social_df)
}

print(results)
