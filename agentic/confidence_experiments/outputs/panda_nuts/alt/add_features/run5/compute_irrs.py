import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = "panda_nuts.csv"
df = pd.read_csv(path)
cols = ["nuts_opened", "seconds", "age", "sex", "help"]
df = df[cols].copy()
mask = df["seconds"].notna() & (df["seconds"] > 0)
mask &= df["nuts_opened"].notna()
mask &= df["age"].notna() & df["sex"].notna() & df["help"].notna()
df = df[mask].copy()

formula = "nuts_opened ~ age + C(sex) + C(help)"
nb_model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df["seconds"])
).fit()

params = nb_model.params
conf = nb_model.conf_int()

irr = np.exp(params)
conf_irr = np.exp(conf)

summary = pd.DataFrame({
    "coef": params,
    "irr": irr,
    "pvalue": nb_model.pvalues,
    "ci_low": conf_irr[0],
    "ci_high": conf_irr[1]
})
print(summary)
