import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Feature engineering: relative group size and contest location
# Positive size_diff means focal group is larger
# Positive dist_diff means focal is closer to its home range center (other is farther)
df["size_diff"] = df["n_focal"] - df["n_other"]
df["dist_diff"] = df["dist_other"] - df["dist_focal"]

# Logistic regression: win ~ size_diff + dist_diff
X = df[["size_diff", "dist_diff"]]
X = sm.add_constant(X)
y = df["win"]

model = sm.Logit(y, X).fit(disp=False)

# Summarize results
summary = model.summary2()
print(summary)

# Save key outputs for quick inspection
params = model.params
pvalues = model.pvalues
odds_ratios = params.apply(lambda x: float(np.exp(x)))

results = pd.DataFrame({
    "coef": params,
    "pvalue": pvalues,
    "odds_ratio": odds_ratios
})
print("\nKey results (coef, pvalue, odds_ratio):")
print(results)

# Also compute simple win rates by size and location advantage
# Size advantage
size_adv = df["size_diff"] > 0
loc_adv = df["dist_diff"] > 0

win_rate_size_adv = df.loc[size_adv, "win"].mean()
win_rate_size_disadv = df.loc[~size_adv, "win"].mean()
win_rate_loc_adv = df.loc[loc_adv, "win"].mean()
win_rate_loc_disadv = df.loc[~loc_adv, "win"].mean()

print("\nWin rates:")
print(f"Size advantage win rate: {win_rate_size_adv:.3f}")
print(f"Size disadvantage win rate: {win_rate_size_disadv:.3f}")
print(f"Location advantage (closer to home) win rate: {win_rate_loc_adv:.3f}")
print(f"Location disadvantage win rate: {win_rate_loc_disadv:.3f}")

# Save summary to a CSV for reference
results.to_csv("analysis_results.csv", index=True)
