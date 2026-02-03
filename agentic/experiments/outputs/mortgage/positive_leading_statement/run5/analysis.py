import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
DF_PATH = 'mortgage.csv'
df = pd.read_csv(DF_PATH)

# Basic check
print("Rows:", len(df))

# Approval/denial by gender
# female: 1 female, 0 male
# accept: 1 accepted, 0 denied
ct = pd.crosstab(df['female'], df['accept'])
ct.index = ['male(0)', 'female(1)']
ct.columns = ['denied(0)', 'accepted(1)']
print("\nContingency table (female x accept):")
print(ct)

# Rates
rates = df.groupby('female')['accept'].mean()
print("\nAcceptance rates by gender:")
for k, v in rates.items():
    label = 'female' if k == 1 else 'male'
    print(f"  {label}: {v:.4f}")

# Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(ct.values)
print(f"\nChi-square test: chi2={chi2:.4f}, dof={dof}, p={p:.6f}")

# Logistic regression with controls
# Outcome: deny (1 denied)
# Use columns that are relevant and present
predictors = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

X = df[predictors].copy()
y = df['deny']

# Drop rows with missing or infinite values in predictors/outcome
model_df = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
X = sm.add_constant(model_df[predictors])
y = model_df['deny']

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

print("\nLogit results (deny ~ predictors):")
print(res.summary2().tables[1].loc[['female']])

# Odds ratio for female
coef = res.params['female']
se = res.bse['female']
odds_ratio = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)
print(f"\nFemale odds ratio (deny): {odds_ratio:.4f} (95% CI {ci_low:.4f}, {ci_high:.4f})")

# Save key results for interpretation
results = {
    'accept_rate_male': rates.get(0.0, np.nan),
    'accept_rate_female': rates.get(1.0, np.nan),
    'chi2_p': p,
    'female_logit_coef': coef,
    'female_logit_p': res.pvalues['female'],
    'female_odds_ratio': odds_ratio,
    'female_or_ci_low': ci_low,
    'female_or_ci_high': ci_high,
}
print("\nKey results:")
for k, v in results.items():
    print(f"  {k}: {v}")
