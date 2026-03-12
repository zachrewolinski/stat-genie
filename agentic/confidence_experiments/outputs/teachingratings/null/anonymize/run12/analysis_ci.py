import pandas as pd
import statsmodels.formula.api as smf

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
    'feature13': 'instructor_id',
    'feature1': 'course_id'
}

df = pd.read_csv('teachingratings.csv').rename(columns=rename_map)

formula_controls = (
    'rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) '
    '+ C(upper_div) + C(native_english) + C(tenure_track) '
    '+ students_participated + students_enrolled'
)

model_controls = smf.ols(formula_controls, data=df).fit(cov_type='HC3')

ci = model_controls.conf_int().loc['beauty']
print({'coef': model_controls.params['beauty'], 'p': model_controls.pvalues['beauty'], 'ci_low': ci[0], 'ci_high': ci[1]})
