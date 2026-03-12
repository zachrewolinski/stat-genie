import pandas as pd
import statsmodels.api as sm

DF = pd.read_csv("crofoot.csv")
rel_size = DF["feature7"] - DF["feature8"]
loc_diff = DF["feature5"] - DF["feature6"]
outcome = DF["feature4"]
X = sm.add_constant(pd.DataFrame({"rel_size": rel_size, "loc_diff": loc_diff}))
model = sm.Logit(outcome, X).fit(disp=False)
print(model.summary())
print(model.conf_int())
