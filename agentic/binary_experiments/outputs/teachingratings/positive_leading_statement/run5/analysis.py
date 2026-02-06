import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'teachingratings.csv'
df = pd.read_csv(DF_PATH)

# Basic OLS with controls
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + "
    "C(division) + C(native) + C(tenure) + students + allstudents"
)
model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Also a simple bivariate model for intuition
biv = smf.ols("eval ~ beauty", data=df).fit(cov_type='HC3')

# Print key results
print("Bivariate model (robust SE):")
print({"coef": biv.params['beauty'], "pvalue": biv.pvalues['beauty'], "ci": biv.conf_int().loc['beauty'].tolist()})

print("\nControlled model (robust SE):")
print({"coef": model.params['beauty'], "pvalue": model.pvalues['beauty'], "ci": model.conf_int().loc['beauty'].tolist()})

# Save results for conclusion
results = {
    "bivariate_coef": float(biv.params['beauty']),
    "bivariate_p": float(biv.pvalues['beauty']),
    "bivariate_ci": [float(x) for x in biv.conf_int().loc['beauty'].tolist()],
    "controlled_coef": float(model.params['beauty']),
    "controlled_p": float(model.pvalues['beauty']),
    "controlled_ci": [float(x) for x in model.conf_int().loc['beauty'].tolist()],
}

pd.Series(results).to_json('analysis_results.json')
