import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('boxes.csv')

# Keep only relevant columns and drop missing
cols = ['y', 'age', 'culture']
df = _df[cols].dropna().copy()

# Define outcomes
# y: 1=unchosen option, 2=majority option, 3=minority option
# Social information reliance: choosing demonstrated options (majority or minority)
df['choice_demo'] = df['y'].isin([2, 3]).astype(int)
# Majority preference among all choices
_df['choice_majority'] = (_df['y'] == 2).astype(int)
# Majority preference among demonstrated choices only
sub_demo = df[df['y'].isin([2, 3])].copy()
sub_demo['choice_majority'] = (sub_demo['y'] == 2).astype(int)

# Descriptives
overall = {
    'n_total': len(df),
    'majority_rate_all': (df['y'] == 2).mean(),
    'minority_rate_all': (df['y'] == 3).mean(),
    'unchosen_rate_all': (df['y'] == 1).mean(),
    'demo_rate': df['choice_demo'].mean(),
    'majority_rate_demo': sub_demo['choice_majority'].mean() if len(sub_demo) else np.nan,
}

# Logistic regression: reliance on social info
# choice_demo ~ age + culture (categorical)
model_demo = smf.logit('choice_demo ~ age + C(culture)', data=df).fit(disp=False)

# Logistic regression: preference for majority among demonstrated choices
model_majority = smf.logit('choice_majority ~ age + C(culture)', data=sub_demo).fit(disp=False)

# Extract key p-values
p_age_demo = model_demo.pvalues.get('age', np.nan)
p_age_majority = model_majority.pvalues.get('age', np.nan)

# Any culture effect?
# Consider any culture coefficient significant at 0.05
culture_terms_demo = [k for k in model_demo.pvalues.index if k.startswith('C(culture)')]
culture_terms_majority = [k for k in model_majority.pvalues.index if k.startswith('C(culture)')]

p_culture_demo = model_demo.pvalues[culture_terms_demo].min() if culture_terms_demo else np.nan
p_culture_majority = model_majority.pvalues[culture_terms_majority].min() if culture_terms_majority else np.nan

# Odds ratios for age
or_age_demo = float(np.exp(model_demo.params['age']))
or_age_majority = float(np.exp(model_majority.params['age']))

print('Descriptives:', overall)
print('Demo model p(age):', p_age_demo, 'OR(age):', or_age_demo)
print('Demo model min p(culture):', p_culture_demo)
print('Majority model p(age):', p_age_majority, 'OR(age):', or_age_majority)
print('Majority model min p(culture):', p_culture_majority)

# Save key results for interpretation
results = {
    'overall': overall,
    'p_age_demo': float(p_age_demo),
    'p_age_majority': float(p_age_majority),
    'p_culture_demo': float(p_culture_demo),
    'p_culture_majority': float(p_culture_majority),
    'or_age_demo': or_age_demo,
    'or_age_majority': or_age_majority,
}

pd.Series(results).to_json('analysis_results.json')
