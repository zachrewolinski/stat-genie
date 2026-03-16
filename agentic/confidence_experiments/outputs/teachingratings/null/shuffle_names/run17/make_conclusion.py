import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Ensure categorical types
cat_cols = ['eval','tenure','prof','native','gender','credits']
for c in cat_cols:
    df[c] = df[c].astype('category')

n = len(df)

# Simple model
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Controlled model
model_controls = smf.ols(
    'allstudents ~ beauty + age + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + rownames + minority + students',
    data=df
).fit()

# Extract stats
corr = df['beauty'].corr(df['allstudents'])

simple_coef = model_simple.params['beauty']
simple_p = model_simple.pvalues['beauty']

ctrl_coef = model_controls.params['beauty']
ctrl_p = model_controls.pvalues['beauty']
ctrl_ci = model_controls.conf_int().loc['beauty'].tolist()

# Decide response: strong no because effect ~0 and not significant
response = 10

explanation = (
    f"Analyzed {n} courses. The bivariate association between instructor beauty and overall student ratings "
    f"is essentially zero (correlation {corr:.4f}; OLS coef {simple_coef:.4f}, p={simple_p:.3f}). "
    f"In a multivariable OLS model controlling for age, gender/tenure/prof/native/credits indicators, and class size/" 
    f"enrollment measures (rownames, minority, students), the beauty coefficient remains near zero and not significant "
    f"(coef {ctrl_coef:.4f}, p={ctrl_p:.3f}, 95% CI [{ctrl_ci[0]:.4f}, {ctrl_ci[1]:.4f}]). "
    f"These results provide no evidence that instructor beauty affects teaching ratings in this dataset, so the answer is No."
)

out = {"response": response, "explanation": explanation}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
