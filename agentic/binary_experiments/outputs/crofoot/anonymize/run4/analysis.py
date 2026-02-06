import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
csv_path = "crofoot.csv"
df = pd.read_csv(csv_path)

# Create predictors
# Relative group size: focal group size - other group size
# Contest location: other distance - focal distance (positive means contest closer to focal group's center)
df["rel_group_size"] = df["feature7"] - df["feature8"]
df["rel_location"] = df["feature6"] - df["feature5"]

# Outcome
y = df["feature4"]

# Model: win ~ relative group size + contest location
X = df[["rel_group_size", "rel_location"]]
X = sm.add_constant(X)

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()

print(result.summary())

# Also report odds ratios for interpretability
params = result.params
conf = result.conf_int()
odds_ratios = params.apply(lambda v: float(np.exp(v)))
conf_odds = conf.applymap(lambda v: float(np.exp(v)))

print("\nOdds ratios:")
for name in params.index:
    print(f"{name}: OR={odds_ratios[name]:.3f}, 95% CI=({conf_odds.loc[name, 0]:.3f}, {conf_odds.loc[name, 1]:.3f}), p={result.pvalues[name]:.4f}")
