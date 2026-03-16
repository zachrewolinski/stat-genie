import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
file_path = "teachingratings.csv"
df = pd.read_csv(file_path)

# Rename for readability
# Map features to semantic names
rename_map = {
    'feature2': 'minority',
    'feature3': 'age',
    'feature4': 'gender',
    'feature5': 'single_credit',
    'feature6': 'beauty',
    'feature7': 'rating',
    'feature8': 'upper_div',
    'feature9': 'native_english',
    'feature10': 'tenure_track',
    'feature11': 'students_participated',
    'feature12': 'students_enrolled',
    'feature13': 'instructor_id'
}

# Keep feature1 as course_id for completeness
rename_map['feature1'] = 'course_id'

df = df.rename(columns=rename_map)

# Basic correlation between beauty and rating
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['rating'])

# Simple OLS: rating ~ beauty
model_simple = smf.ols('rating ~ beauty', data=df).fit()

# OLS with controls (categorical and numeric)
# Use C() for categorical
formula_controls = (
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) '
    '+ C(upper_div) + C(native_english) + C(tenure_track) '
    '+ students_participated + students_enrolled'
)
model_controls = smf.ols(formula_controls, data=df).fit(cov_type='HC3')

# Standardized effect: z-score for beauty and rating
# Compute standardized coefficients using z-scores for numeric variables
# We'll fit a model with standardized beauty and rating (others included standardized as well)
# But for inference we'll focus on beauty's standardized coefficient
numeric_cols = ['beauty', 'age', 'students_participated', 'students_enrolled']
df_std = df.copy()
for col in numeric_cols:
    df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std(ddof=0)

df_std['rating'] = (df_std['rating'] - df_std['rating'].mean()) / df_std['rating'].std(ddof=0)

model_std = smf.ols(formula_controls.replace('rating', 'rating'), data=df_std).fit(cov_type='HC3')

# Collect results
results = {
    'n': len(df),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'controls_coef': model_controls.params['beauty'],
    'controls_p': model_controls.pvalues['beauty'],
    'controls_r2': model_controls.rsquared,
    'std_coef': model_std.params['beauty'],
    'std_p': model_std.pvalues['beauty']
}

print(results)

