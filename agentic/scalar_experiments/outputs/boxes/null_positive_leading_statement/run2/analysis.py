import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('boxes.csv')

# Basic sanity checks
n = len(df)

# Majority choice indicator (2 = majority)
df['majority_choice'] = (df['y'] == 2).astype(int)

# Overall majority choice rate
overall_rate = df['majority_choice'].mean()

# Majority rate by age (treated as continuous, but also group for descriptives)
age_bins = pd.cut(df['age'], bins=[3, 6, 9, 12, 15], labels=['4-6', '7-9', '10-12', '13-14'])
age_rate = df.groupby(age_bins)['majority_choice'].mean()

# Majority rate by culture
culture_rate = df.groupby('culture')['majority_choice'].mean()

# Logistic regression: majority_choice ~ age + C(culture) + gender + majority_first
# Use one culture as baseline via categorical coding
formula = 'majority_choice ~ age + C(culture) + gender + majority_first'
logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

# Add interaction of age and culture to test developmental variation across cultures
formula_int = 'majority_choice ~ age * C(culture) + gender + majority_first'
logit_int_model = smf.logit(formula=formula_int, data=df).fit(disp=False)

# Likelihood ratio test comparing models with and without interaction
lr_stat = 2 * (logit_int_model.llf - logit_model.llf)
lr_df = logit_int_model.df_model - logit_model.df_model
lr_pvalue = sm.stats.chisqprob(lr_stat, lr_df) if hasattr(sm.stats, 'chisqprob') else 1.0

# Extract key summaries
age_coef = logit_model.params.get('age', np.nan)
age_p = logit_model.pvalues.get('age', np.nan)

culture_ps = logit_model.pvalues[[c for c in logit_model.pvalues.index if c.startswith('C(culture)')]]

# Summarize evidence strength heuristically to guide Likert score

summary = {
    'n': int(n),
    'overall_majority_rate': float(overall_rate),
    'age_rate_by_bin': age_rate.to_dict(),
    'culture_rate': culture_rate.to_dict(),
    'age_coef': float(age_coef) if np.isfinite(age_coef) else None,
    'age_p': float(age_p) if np.isfinite(age_p) else None,
    'culture_pvalues': {k: float(v) for k, v in culture_ps.items()},
    'lr_pvalue_age_by_culture': float(lr_pvalue) if np.isfinite(lr_pvalue) else None,
}

print('SUMMARY_START')
print(summary)
print('SUMMARY_END')
