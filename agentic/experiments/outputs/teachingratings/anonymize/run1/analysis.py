import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Rename columns for clarity
rename_map = {
    "feature1": "course_id",
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "rating",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure_track",
    "feature11": "students_eval",
    "feature12": "students_enroll",
    "feature13": "instructor_id",
}

df = df.rename(columns=rename_map)

# Basic correlation
corr = df["beauty"].corr(df["rating"])
print(f"Correlation between beauty and rating: {corr:.4f}")

# OLS with controls; use robust (HC3) standard errors
formula = (
    "rating ~ beauty + age + C(gender) + C(minority) + C(single_credit) + "
    "C(division) + C(native_english) + C(tenure_track) + students_eval + students_enroll"
)

model = smf.ols(formula, data=df).fit(cov_type="HC3")
print(model.summary())

beauty_coef = model.params["beauty"]
beauty_p = model.pvalues["beauty"]
beauty_sd = df["beauty"].std()

print(f"Beauty coefficient: {beauty_coef:.4f}")
print(f"Beauty p-value: {beauty_p:.4g}")
print(f"Effect per 1 SD beauty: {beauty_coef * beauty_sd:.4f} rating points")
