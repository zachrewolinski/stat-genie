import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic checks
n = len(df)

# Prepare variables
# Use log class size to reduce skew
if 'allstudents' in df.columns:
    df['log_allstudents'] = np.log(df['allstudents'])

# Simple regression
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Controlled regression
formula_controls = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + "
    "C(division) + C(native) + C(tenure) + log_allstudents"
)
model_controls = smf.ols(formula_controls, data=df).fit(cov_type="HC3")

# Correlation
corr = df['beauty'].corr(df['eval'])

# Standardized effect for beauty (simple and controls)
sd_beauty = df['beauty'].std()
_sd_eval = df['eval'].std()
std_beta_simple = model_simple.params['beauty'] * sd_beauty / _sd_eval
std_beta_controls = model_controls.params['beauty'] * sd_beauty / _sd_eval

results = {
    "n": n,
    "corr_beauty_eval": corr,
    "simple_coef": model_simple.params['beauty'],
    "simple_p": model_simple.pvalues['beauty'],
    "simple_ci": model_simple.conf_int().loc['beauty'].tolist(),
    "simple_r2": model_simple.rsquared,
    "simple_std_beta": std_beta_simple,
    "controls_coef": model_controls.params['beauty'],
    "controls_p": model_controls.pvalues['beauty'],
    "controls_ci": model_controls.conf_int().loc['beauty'].tolist(),
    "controls_r2": model_controls.rsquared,
    "controls_std_beta": std_beta_controls,
}

print(results)
