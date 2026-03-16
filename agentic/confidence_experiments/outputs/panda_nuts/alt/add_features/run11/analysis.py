import json
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
DATA_PATH = 'panda_nuts.csv'
df = pd.read_csv(DATA_PATH)

# Keep relevant columns and drop missing
cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help']
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

sub = df[cols].copy()
# Normalize categories
sub['sex'] = sub['sex'].astype(str).str.strip().str.lower()
sub['help'] = sub['help'].astype(str).str.strip().str.lower()

# Remove rows with nonpositive seconds
sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
sub = sub[sub['seconds'] > 0]

# Build Poisson GLM with log(seconds) offset
sub['log_seconds'] = np.log(sub['seconds'])

model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=sub,
    family=sm.families.Poisson(),
    offset=sub['log_seconds']
)
res = model.fit(cov_type='HC1')

# Overdispersion check
mu = res.mu
pearson_chi2 = np.sum((sub['nuts_opened'] - mu) ** 2 / mu)
df_resid = res.df_resid
overdispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

# Negative Binomial as sensitivity if overdispersion > ~1.5
nb_res = None
if np.isfinite(overdispersion) and overdispersion > 1.5:
    nb_model = smf.glm(
        formula='nuts_opened ~ age + C(sex) + C(help)',
        data=sub,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=sub['log_seconds']
    )
    nb_res = nb_model.fit(cov_type='HC1')

# Rate-based OLS sensitivity
sub['rate'] = sub['nuts_opened'] / sub['seconds']
ols = smf.ols('rate ~ age + C(sex) + C(help)', data=sub).fit(cov_type='HC1')

# Extract p-values for key predictors
# For categorical, evaluate overall effect via Wald test
sex_terms = [t for t in res.params.index if t.startswith('C(sex)')]
help_terms = [t for t in res.params.index if t.startswith('C(help)')]

def wald_pvalue(result, terms):
    if not terms:
        return None
    if len(terms) == 1:
        constraint = f"{terms[0]} = 0"
    else:
        constraint = ", ".join([f"{t} = 0" for t in terms])
    test = result.wald_test(constraint)
    return float(test.pvalue)

# Wald tests
sex_p = wald_pvalue(res, sex_terms)
help_p = wald_pvalue(res, help_terms)

age_p = float(res.pvalues.get('age', np.nan))

# Sensitivity p-values from OLS
ols_age_p = float(ols.pvalues.get('age', np.nan))
ols_sex_p = None
ols_help_p = None
if sex_terms:
    # use OLS terms to wald test too
    ols_sex_terms = [t for t in ols.params.index if t.startswith('C(sex)')]
    if ols_sex_terms:
        ols_sex_p = wald_pvalue(ols, ols_sex_terms)
if help_terms:
    ols_help_terms = [t for t in ols.params.index if t.startswith('C(help)')]
    if ols_help_terms:
        ols_help_p = wald_pvalue(ols, ols_help_terms)

# Determine evidence
alpha = 0.05
sig_age = age_p < alpha
sig_sex = (sex_p is not None) and (sex_p < alpha)
sig_help = (help_p is not None) and (help_p < alpha)

# If NB model exists, check whether conclusions are consistent
nb_note = None
if nb_res is not None:
    nb_age_p = float(nb_res.pvalues.get('age', np.nan))
    nb_sex_terms = [t for t in nb_res.params.index if t.startswith('C(sex)')]
    nb_help_terms = [t for t in nb_res.params.index if t.startswith('C(help)')]
    nb_sex_p = None
    nb_help_p = None
    if nb_sex_terms:
        nb_sex_p = wald_pvalue(nb_res, nb_sex_terms)
    if nb_help_terms:
        nb_help_p = wald_pvalue(nb_res, nb_help_terms)
    nb_note = {
        'age_p': nb_age_p,
        'sex_p': nb_sex_p,
        'help_p': nb_help_p,
    }

# Compute effect sizes for interpretation (rate ratios)
params = res.params
conf = res.conf_int()
rate_ratios = np.exp(params)
conf_rr = np.exp(conf)

# Build explanation
n = len(sub)

explanation_parts = []
explanation_parts.append(
    f"Modeled nut-cracking efficiency as nuts opened per second using a Poisson regression with log(seconds) offset (n={n})."
)
explanation_parts.append(
    f"Overdispersion ratio (Pearson chi2/df) was {overdispersion:.2f}; {'also fit a Negative Binomial as sensitivity.' if nb_res is not None else 'Poisson used as primary model.'}"
)

# Describe effects
age_rr = rate_ratios.get('age', np.nan)
age_ci = conf_rr.loc['age'].tolist() if 'age' in conf_rr.index else [np.nan, np.nan]
explanation_parts.append(
    f"Age effect (rate ratio per year) = {age_rr:.3f} (95% CI {age_ci[0]:.3f} to {age_ci[1]:.3f}), p={age_p:.4f}."
)

# For sex and help, take base level from model; report overall p-values
if sex_p is not None:
    explanation_parts.append(f"Sex overall effect Wald p={sex_p:.4f}.")
if help_p is not None:
    explanation_parts.append(f"Help overall effect Wald p={help_p:.4f}.")

# Sensitivity with OLS
explanation_parts.append(
    f"OLS on rate gave p-values: age={ols_age_p:.4f}, sex={ols_sex_p:.4f}, help={ols_help_p:.4f}."
)

if nb_note is not None:
    explanation_parts.append(
        f"Negative Binomial sensitivity p-values: age={nb_note['age_p']:.4f}, sex={nb_note['sex_p']:.4f}, help={nb_note['help_p']:.4f}."
    )

# Decide Likert response
# If at least two predictors significant with consistent direction, strong yes; if one significant weak yes; if none, no.
num_sig = sum([sig_age, sig_sex, sig_help])
if num_sig == 0:
    response = 20  # strong no
    conclusion = "No clear evidence that age, sex, or help influence nut-cracking efficiency."
elif num_sig == 1:
    response = 60  # modest yes
    conclusion = "Some evidence that at least one of age, sex, or help influences efficiency, but not all are significant."
elif num_sig == 2:
    response = 75  # yes
    conclusion = "Evidence that multiple factors (age/sex/help) influence efficiency."
else:
    response = 85  # strong yes
    conclusion = "Strong evidence that age, sex, and help all influence efficiency."

explanation_parts.append(conclusion)

explanation = " ".join(explanation_parts)

out = {"response": int(response), "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)

# Print summary to stdout for debugging
print(json.dumps(out, indent=2))
