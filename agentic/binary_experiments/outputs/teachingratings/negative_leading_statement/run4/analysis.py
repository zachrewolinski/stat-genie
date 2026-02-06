import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = "teachingratings.csv"
df = pd.read_csv(DF_PATH)

# Basic checks
print("Rows, columns:", df.shape)
print("Missing values:\n", df.isna().sum())

# Simple association
corr = df["eval"].corr(df["beauty"])
print(f"Correlation eval~beauty: {corr:.3f}")

# Model 1: bivariate
m1 = smf.ols("eval ~ beauty", data=df).fit()
print("\nModel 1: eval ~ beauty")
print(m1.summary().tables[1])

# Model 2: add controls (robust SE)
formula = (
    "eval ~ beauty + age + C(gender) + C(minority) + C(credits) "
    "+ C(division) + C(native) + C(tenure) + students + allstudents"
)
m2 = smf.ols(formula, data=df).fit(cov_type="HC3")
print("\nModel 2: eval ~ beauty + controls (HC3 robust SE)")
print(m2.summary().tables[1])

# Effect size interpretation: change in eval for +1 SD beauty
beauty_sd = df["beauty"].std()
coef = m2.params["beauty"]
print(f"\nBeauty SD: {beauty_sd:.3f}")
print(f"Model 2 beauty coef: {coef:.3f}")
print(f"Eval change for +1 SD beauty: {coef * beauty_sd:.3f} points")
