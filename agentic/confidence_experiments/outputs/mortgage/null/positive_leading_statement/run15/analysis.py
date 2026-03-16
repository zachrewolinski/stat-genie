import json
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
df = pd.read_csv('mortgage.csv')

# Ensure expected columns
# We'll use 'accept' as outcome (1 accepted, 0 denied)
# Predictor of interest: 'female' (1 female, 0 male)

# Drop rows with missing values in variables used
cols = [
    'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

# Some datasets might not include all controls; keep only existing
cols = [c for c in cols if c in df.columns]

analysis_df = df[cols].dropna().copy()

# Basic counts and approval rates by gender
summary = analysis_df.groupby('female')['accept'].agg(['mean', 'count'])

# Logistic regression: accept ~ female + controls
X = analysis_df.drop(columns=['accept'])
X = sm.add_constant(X, has_constant='add')

y = analysis_df['accept']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Extract female coefficient and p-value
if 'female' in result.params.index:
    coef = float(result.params['female'])
    pval = float(result.pvalues['female'])
    odds_ratio = float(np.exp(coef))
else:
    coef = pval = odds_ratio = np.nan

# Unadjusted difference in approval rates
if 0.0 in summary.index and 1.0 in summary.index:
    rate_male = float(summary.loc[0.0, 'mean'])
    rate_female = float(summary.loc[1.0, 'mean'])
else:
    rate_male = rate_female = np.nan

# Decide Likert response based on significance and effect size
# If p-value >= 0.05, lean "No"; if significant, lean "Yes" with magnitude
if np.isnan(pval):
    response = 50
    conclusion = "Gender effect could not be estimated in the model."
else:
    if pval < 0.01:
        # stronger evidence
        if odds_ratio > 1:
            response = 70
        elif odds_ratio < 1:
            response = 30
        else:
            response = 50
    elif pval < 0.05:
        # moderate evidence
        if odds_ratio > 1:
            response = 60
        elif odds_ratio < 1:
            response = 40
        else:
            response = 50
    else:
        # not significant
        response = 35

# Build explanation
explanation = (
    f"We tested whether gender (female=1) predicts mortgage approval (accept=1). "
    f"Unadjusted approval rates: male={rate_male:.3f}, female={rate_female:.3f} "
    f"(n male={int(summary.loc[0.0, 'count']) if 0.0 in summary.index else 'NA'}, "
    f"n female={int(summary.loc[1.0, 'count']) if 1.0 in summary.index else 'NA'}). "
    f"A logistic regression controlling for creditworthiness and application factors "
    f"(black, housing_expense_ratio, self_employed, married, mortgage_credit, consumer_credit, "
    f"bad_history, PI_ratio, loan_to_value, denied_PMI) produced a female coefficient of {coef:.3f} "
    f"(odds ratio {odds_ratio:.3f}, p={pval:.4f}). "
)

if pval >= 0.05:
    explanation += (
        "The p-value indicates no statistically significant evidence that gender affects approval once "
        "other factors are accounted for. This supports a 'No' conclusion on the effect of gender."
    )
else:
    if odds_ratio > 1:
        explanation += (
            "The positive, statistically significant coefficient suggests female applicants have higher odds "
            "of approval after controls, supporting a 'Yes' (gender affects approval) with a modest effect."
        )
    elif odds_ratio < 1:
        explanation += (
            "The negative, statistically significant coefficient suggests female applicants have lower odds "
            "of approval after controls, supporting a 'Yes' (gender affects approval) with a modest effect."
        )
    else:
        explanation += (
            "The coefficient is near zero despite significance; practical effect appears minimal."
        )

# Write conclusion.txt
with open('conclusion.txt', 'w') as f:
    json.dump({'response': int(response), 'explanation': explanation}, f)

print('Wrote conclusion.txt')
