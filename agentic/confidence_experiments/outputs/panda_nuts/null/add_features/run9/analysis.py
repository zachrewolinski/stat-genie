import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.discrete.discrete_model as smd


df = pd.read_csv('panda_nuts.csv')

# Define efficiency as nuts opened per second.
df['efficiency'] = df['nuts_opened'] / df['seconds']

# OLS on efficiency (continuous rate)
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()

# Count model with exposure (seconds) to estimate rate effects robustly.
df['log_seconds'] = np.log(df['seconds'])
exog = sm.add_constant(pd.get_dummies(df[['age', 'sex', 'help']], drop_first=True))
nb2 = smd.NegativeBinomial(df['nuts_opened'], exog, offset=df['log_seconds']).fit(disp=False)

# Poisson overdispersion check (not used for inference, just diagnostic)
poisson = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=df['log_seconds'],
).fit()
mu = poisson.mu
pearson_chi2 = ((df['nuts_opened'] - mu) ** 2 / mu).sum()
overdisp_ratio = pearson_chi2 / poisson.df_resid

# Extract key stats
n = len(df)

ols_p = ols.pvalues
ols_r2 = ols.rsquared

nb2_p = nb2.pvalues
nb2_llr_p = nb2.llr_pvalue
nb2_pr2 = nb2.prsquared

# Incidence rate ratios (IRR) and 95% CI for NB2
params = nb2.params
conf = nb2.conf_int()
irr = np.exp(params)
conf_irr = np.exp(conf)

# Build explanation
explanation = (
    f"Analyzed {n} chimpanzee sessions. Efficiency was defined as nuts_opened/seconds; "
    f"OLS on efficiency showed no evidence of effects (age p={ols_p['age']:.3f}, "
    f"sex p={ols_p['C(sex)[T.m]']:.3f}, help p={ols_p['C(help)[T.y]']:.3f}; R^2={ols_r2:.3f}). "
    f"Because nut counts were highly overdispersed for a Poisson rate model (Pearson chi^2/df ≈ {overdisp_ratio:.1f}), "
    f"I fit a negative binomial rate model with log(seconds) as an offset. That model was not significant overall (LLR p={nb2_llr_p:.3f}, "
    f"pseudo R^2={nb2_pr2:.3f}) and none of the predictors were significant: "
    f"age IRR={irr['age']:.3f} (95% CI {conf_irr.loc['age',0]:.3f}-{conf_irr.loc['age',1]:.3f}, p={nb2_p['age']:.3f}), "
    f"male vs female IRR={irr['sex_m']:.3f} (95% CI {conf_irr.loc['sex_m',0]:.3f}-{conf_irr.loc['sex_m',1]:.3f}, p={nb2_p['sex_m']:.3f}), "
    f"help vs no-help IRR={irr['help_y']:.3f} (95% CI {conf_irr.loc['help_y',0]:.3f}-{conf_irr.loc['help_y',1]:.3f}, p={nb2_p['help_y']:.3f}). "
    "Overall, the data do not provide statistically significant evidence that age, sex, or receiving help influences nut-cracking efficiency."
)

result = {
    "response": 30,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
