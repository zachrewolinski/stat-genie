import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Basic cleaning: ensure binary columns are numeric 0/1

# Outcome: deny (1 denied). We'll also compute accept.

# Descriptive stats

n_total = len(df)

# group rates

group_rates = df.groupby('female')['deny'].agg(['mean','count'])

# Chi-square test of independence
contingency = pd.crosstab(df['female'], df['deny'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)

# Logistic regression unadjusted
model_unadj = smf.logit('deny ~ female', data=df).fit(disp=False)

# Adjusted model with covariates
# Avoid including outcome-related variables that are essentially derived from decision? We'll exclude 'accept' (perfect inverse) and 'deny' itself.
# Include creditworthiness/financial characteristics; exclude denied_PMI to avoid post-decision signal.

covariates = [
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value'
]

formula = 'deny ~ ' + ' + '.join(covariates)
model_adj = smf.logit(formula, data=df).fit(disp=False)

# Extract female effect

def summarize_female(model):
    coef = model.params['female']
    se = model.bse['female']
    pval = model.pvalues['female']
    # odds ratio and 95% CI
    or_val = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    return {
        'coef': coef,
        'se': se,
        'pval': pval,
        'odds_ratio': or_val,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

summary_unadj = summarize_female(model_unadj)
summary_adj = summarize_female(model_adj)

# Also calculate marginal difference in denial rates between female and male

rate_male = group_rates.loc[0, 'mean'] if 0 in group_rates.index else np.nan
rate_female = group_rates.loc[1, 'mean'] if 1 in group_rates.index else np.nan
rate_diff = rate_female - rate_male

# Save key results to a text file for inspection

with open('analysis_results.txt','w') as f:
    f.write(f"Total n: {n_total}\n")
    f.write("Denial rate by female:\n")
    f.write(group_rates.to_string())
    f.write("\n\nChi-square test (female vs deny):\n")
    f.write(f"chi2={chi2:.4f}, p={p_chi:.6g}, dof={dof}\n")
    f.write("\nUnadjusted logit deny~female:\n")
    f.write(str(summary_unadj))
    f.write("\n\nAdjusted logit deny~covariates:\n")
    f.write(str(summary_adj))
    f.write("\n\nRate difference female-male: " + str(rate_diff) + "\n")
    f.write("\nModel summaries:\n")
    f.write(model_unadj.summary().as_text())
    f.write("\n\n" + model_adj.summary().as_text())

print('Done')
