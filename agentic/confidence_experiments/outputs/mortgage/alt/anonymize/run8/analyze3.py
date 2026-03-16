import pandas as pd
import statsmodels.api as sm

_df = pd.read_csv("mortgage.csv")

# Drop missing in gender or outcome
_df = _df.dropna(subset=["feature2", "feature14"])
X = sm.add_constant(_df[["feature2"]])
y = _df["feature14"]
res = sm.Logit(y, X).fit(disp=False)
print(res.summary2().tables[1])
