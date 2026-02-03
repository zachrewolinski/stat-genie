import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
DF_PATH = "mortgage.csv"
df = pd.read_csv(DF_PATH)

# Key variables
# feature2: 1 if applicant is female, 0 if male
# feature14: 1 if application accepted, 0 if denied

# Descriptive approval rates by gender
approval_by_gender = df.groupby("feature2")["feature14"].mean()
counts_by_gender = df["feature2"].value_counts().sort_index()

# Two-proportion z-test (female vs male)
# p1 = approval rate for females, p0 = approval rate for males
female_mask = df["feature2"] == 1
male_mask = df["feature2"] == 0
p1 = df.loc[female_mask, "feature14"].mean()
p0 = df.loc[male_mask, "feature14"].mean()

n1 = female_mask.sum()
n0 = male_mask.sum()

p_pool = df["feature14"].mean()
se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n0)) ** 0.5
z = (p1 - p0) / se
p_value_z = 2 * (1 - stats.norm.cdf(abs(z)))

# Logistic regression controlling for other factors
# Drop identifier-like feature1 and avoid redundancy with feature11 (denied) and feature14 (accepted)
features = [
    "feature2",  # gender (female=1)
    "feature3",  # Black
    "feature4",  # housing expense ratio
    "feature5",  # self-employed
    "feature6",  # married
    "feature7",  # mortgage credit score
    "feature8",  # consumer credit score
    "feature9",  # bad credit history
    "feature10", # total debt ratio
    "feature12", # loan-to-value
    "feature13", # PMI denied
]

X = df[features].copy()
y = df["feature14"]

# Clean missing or infinite values for modeling
model_df = pd.concat([X, y], axis=1).replace([float("inf"), float("-inf")], pd.NA).dropna()
X = sm.add_constant(model_df[features])
y = model_df["feature14"]

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit(disp=False)

coef_gender = logit_result.params["feature2"]
p_value_gender = logit_result.pvalues["feature2"]

# Save key outputs for conclusion
results_summary = {
    "approval_rate_male": float(p0),
    "approval_rate_female": float(p1),
    "z_test_p_value": float(p_value_z),
    "logit_gender_coef": float(coef_gender),
    "logit_gender_p_value": float(p_value_gender),
}

print("Approval rates by gender (0=male, 1=female):")
print(approval_by_gender)
print("Counts by gender:")
print(counts_by_gender)
print("Two-proportion z-test p-value:", p_value_z)
print("Logit gender coef:", coef_gender)
print("Logit gender p-value:", p_value_gender)

# Write results for downstream use
pd.Series(results_summary).to_csv("analysis_results.csv")
