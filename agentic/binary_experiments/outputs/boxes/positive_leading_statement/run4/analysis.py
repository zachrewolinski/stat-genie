import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ALPHA = 0.05

# Load data

df = pd.read_csv('boxes.csv')

# Social reliance: chose majority or minority vs unchosen

df['social'] = (df['y'] != 1).astype(int)

full_social = smf.logit('social ~ age + C(culture) + gender + majority_first', data=df).fit(disp=0)
red_social_age = smf.logit('social ~ C(culture) + gender + majority_first', data=df).fit(disp=0)
red_social_cul = smf.logit('social ~ age + gender + majority_first', data=df).fit(disp=0)

lr_social_age = 2 * (full_social.llf - red_social_age.llf)
ldf_social_age = full_social.df_model - red_social_age.df_model
p_social_age = stats.chi2.sf(lr_social_age, ldf_social_age)

lr_social_cul = 2 * (full_social.llf - red_social_cul.llf)
ldf_social_cul = full_social.df_model - red_social_cul.df_model
p_social_cul = stats.chi2.sf(lr_social_cul, ldf_social_cul)

# Majority preference among those who used social info
sub = df[df['y'] != 1].copy()
sub['majority'] = (sub['y'] == 2).astype(int)

full_majority = smf.logit('majority ~ age + C(culture) + gender + majority_first', data=sub).fit(disp=0)
red_majority_age = smf.logit('majority ~ C(culture) + gender + majority_first', data=sub).fit(disp=0)
red_majority_cul = smf.logit('majority ~ age + gender + majority_first', data=sub).fit(disp=0)

lr_majority_age = 2 * (full_majority.llf - red_majority_age.llf)
ldf_majority_age = full_majority.df_model - red_majority_age.df_model
p_majority_age = stats.chi2.sf(lr_majority_age, ldf_majority_age)

lr_majority_cul = 2 * (full_majority.llf - red_majority_cul.llf)
ldf_majority_cul = full_majority.df_model - red_majority_cul.df_model
p_majority_cul = stats.chi2.sf(lr_majority_cul, ldf_majority_cul)

results = {
    'p_social_age': p_social_age,
    'p_social_culture': p_social_cul,
    'p_majority_age': p_majority_age,
    'p_majority_culture': p_majority_cul,
}

print('Likelihood ratio test p-values')
for k, v in results.items():
    print(f'{k}: {v:.4f}')

# Decision logic: need evidence of variation by BOTH age and culture
# for social reliance and majority preference.

social_varies = (p_social_age < ALPHA) and (p_social_cul < ALPHA)
majority_varies = (p_majority_age < ALPHA) and (p_majority_cul < ALPHA)

answer_yes = social_varies and majority_varies

if answer_yes:
    conclusion = (
        "Yes\n"
        "Both reliance on social information and majority preference show significant variation by age "
        "and by culture based on likelihood ratio tests (p < 0.05)."
    )
else:
    conclusion = (
        "No\n"
        "Likelihood ratio tests do not show significant age and culture effects for both social reliance "
        "and majority preference at p < 0.05 (e.g., p_social_age={:.3f}, p_social_culture={:.3f}, "
        "p_majority_age={:.3f}, p_majority_culture={:.3f})."
    ).format(p_social_age, p_social_cul, p_majority_age, p_majority_cul)

with open('conclusion.txt', 'w') as f:
    f.write(conclusion)
