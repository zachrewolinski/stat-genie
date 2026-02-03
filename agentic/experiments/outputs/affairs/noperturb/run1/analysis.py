import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Create binary outcome: any affair

df["any_affair"] = (df["affairs"] > 0).astype(int)

# Summary by children
summary = df.groupby("children").agg(
    n=("affairs", "size"),
    mean_affairs=("affairs", "mean"),
    median_affairs=("affairs", "median"),
    prop_any_affair=("any_affair", "mean"),
)
print("Summary by children:\n", summary, "\n")

# Difference in means (affairs) and proportions (any affair)
mean_diff = summary.loc["yes", "mean_affairs"] - summary.loc["no", "mean_affairs"]
prop_diff = summary.loc["yes", "prop_any_affair"] - summary.loc["no", "prop_any_affair"]
print(f"Mean affairs difference (yes - no): {mean_diff:.3f}")
print(f"Prop any affair difference (yes - no): {prop_diff:.3f}\n")

# Regression: any affair ~ children + controls
# Controls: yearsmarried, age, religiousness, education, occupation, rating, gender
# Use logistic regression
model = smf.logit(
    "any_affair ~ C(children) + yearsmarried + age + religiousness + education + occupation + rating + C(gender)",
    data=df,
).fit(disp=False)
print(model.summary())

# Extract effect for children yes (reference no)
coef = model.params.get("C(children)[T.yes]", float("nan"))
se = model.bse.get("C(children)[T.yes]", float("nan"))
print(f"\nLogit coef for children=yes: {coef:.3f} (SE {se:.3f})")

# Also run OLS on affairs count (even though censored)
ols = smf.ols(
    "affairs ~ C(children) + yearsmarried + age + religiousness + education + occupation + rating + C(gender)",
    data=df,
).fit()
print("\nOLS on affairs count:")
print(ols.summary())

coef_ols = ols.params.get("C(children)[T.yes]", float("nan"))
se_ols = ols.bse.get("C(children)[T.yes]", float("nan"))
print(f"\nOLS coef for children=yes: {coef_ols:.3f} (SE {se_ols:.3f})")
