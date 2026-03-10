import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Efficiency: nuts opened per second
# Avoid division by zero (duration min 2.5 per metadata)
df['efficiency'] = df['feature5'] / df['feature6']

# Standardize age for interpretability (optional)
df['age'] = df['feature2']

# Encode categorical predictors
# feature3: sex (f/m), feature7: help (y/N)

print('Basic efficiency stats:')
print(df['efficiency'].describe())

# OLS on efficiency
ols_model = smf.ols('efficiency ~ age + C(feature3) + C(feature7)', data=df).fit()
print('\nOLS summary (efficiency):')
print(ols_model.summary())

# Poisson GLM on counts with log(duration) offset to model rate
# Replace zero counts? Poisson handles zeros.
# Use log link with offset log(duration)
df['log_duration'] = np.log(df['feature6'])
poisson_model = smf.glm('feature5 ~ age + C(feature3) + C(feature7)',
                        data=df,
                        family=sm.families.Poisson(),
                        offset=df['log_duration']).fit()
print('\nPoisson GLM summary (count with offset):')
print(poisson_model.summary())

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = poisson_model.pearson_chi2
overdispersion_ratio = pearson_chi2 / poisson_model.df_resid
print(f"\nPoisson overdispersion ratio: {overdispersion_ratio:.3f}")

# If overdispersion, fit Negative Binomial
nb_model = smf.glm('feature5 ~ age + C(feature3) + C(feature7)',
                   data=df,
                   family=sm.families.NegativeBinomial(alpha=1.0),
                   offset=df['log_duration']).fit()
print('\nNegative Binomial GLM summary (count with offset):')
print(nb_model.summary())

# Extract p-values and coefficients for key predictors
for name, model in [('OLS', ols_model), ('Poisson', poisson_model), ('NegBin', nb_model)]:
    print(f"\n{name} coefficients:")
    params = model.params
    pvals = model.pvalues
    for term in params.index:
        if term == 'Intercept':
            continue
        print(f"  {term}: coef={params[term]:.4f}, p={pvals[term]:.4g}")

# Effect sizes for Negative Binomial (rate ratios)
rr = np.exp(nb_model.params)
rr_ci = np.exp(nb_model.conf_int())
print("\nNegative Binomial rate ratios (exp(coef)):")
for term in rr.index:
    if term == 'Intercept':
        continue
    lo, hi = rr_ci.loc[term]
    print(f"  {term}: RR={rr[term]:.3f}, 95% CI=({lo:.3f}, {hi:.3f})")
