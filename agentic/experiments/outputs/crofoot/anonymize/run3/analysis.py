import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DF_PATH = "crofoot.csv"
df = pd.read_csv(DF_PATH)

# Define variables based on metadata
# feature4: 1 if focal group won, 0 otherwise
# feature7: focal group size, feature8: other group size
# feature5: focal distance to home range center, feature6: other distance

df["rel_size"] = df["feature7"] - df["feature8"]
df["rel_loc"] = df["feature5"] - df["feature6"]

# Model 1: relative size and relative location
X1 = sm.add_constant(df[["rel_size", "rel_loc"]])
model1 = sm.Logit(df["feature4"], X1).fit(disp=False)

# Model 2: relative size + both absolute distances (alternate location formulation)
X2 = sm.add_constant(df[["rel_size", "feature5", "feature6"]])
model2 = sm.Logit(df["feature4"], X2).fit(disp=False)

# Summaries
print("Model 1: win ~ relative size + relative location")
print(model1.summary())
print("\nOdds ratios (Model 1):")
print(np.exp(model1.params))

print("\nModel 2: win ~ relative size + focal distance + other distance")
print(model2.summary())
print("\nOdds ratios (Model 2):")
print(np.exp(model2.params))
