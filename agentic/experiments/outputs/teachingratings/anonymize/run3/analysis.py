import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
col_map = {
    "feature2": "minority",
    "feature3": "age",
    "feature4": "gender",
    "feature5": "single_credit",
    "feature6": "beauty",
    "feature7": "rating",
    "feature8": "division",
    "feature9": "native_english",
    "feature10": "tenure",
    "feature11": "n_eval",
    "feature12": "n_enroll",
}
df = df.rename(columns=col_map)

# Outcome and main predictor
y = df["rating"]

# Build controls
controls = [
    "age",
    "n_eval",
    "n_enroll",
    "minority",
    "gender",
    "single_credit",
    "division",
    "native_english",
    "tenure",
]

X = df[["beauty"] + controls].copy()

# One-hot encode categorical controls
X = pd.get_dummies(X, columns=[
    "minority",
    "gender",
    "single_credit",
    "division",
    "native_english",
    "tenure",
], drop_first=True)

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("N:", len(df))
print("Beauty coefficient:", model.params.get("beauty"))
print("Beauty p-value:", model.pvalues.get("beauty"))
print("Beauty 95% CI:", model.conf_int().loc["beauty"].tolist())
print("R-squared:", model.rsquared)

# Also compute simple bivariate correlation
corr = df[["beauty", "rating"]].corr().loc["beauty", "rating"]
print("Correlation(beauty, rating):", corr)

