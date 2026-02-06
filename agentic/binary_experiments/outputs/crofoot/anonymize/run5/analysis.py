import pandas as pd
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Relative group size and contest location
# Relative group size: focal size minus other size
# Contest location: focal distance from its home range center minus other group's distance
# Positive loc_diff means contest occurred farther from focal home center than from other
size_diff = df["feature7"] - df["feature8"]
loc_diff = df["feature5"] - df["feature6"]

X = pd.DataFrame({"size_diff": size_diff, "loc_diff": loc_diff})
X = sm.add_constant(X)
y = df["feature4"]

model = sm.Logit(y, X).fit(disp=False)

print(model.summary())
print("\nCoefficients:\n", model.params)
print("\nP-values:\n", model.pvalues)
