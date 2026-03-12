import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Compute efficiency (nuts opened per second)
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Basic stats
n_rows = len(df)
num_chimps = df["chimpanzee"].nunique()

# Group means
mean_by_sex = df.groupby("sex")["efficiency"].mean()
mean_by_help = df.groupby("help")["efficiency"].mean()

# Correlation with age
age_corr = df[["age", "efficiency"]].corr().iloc[0, 1]

# OLS with cluster-robust SE by chimpanzee (repeated measures)
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df)
res = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

# Collect key results
coef = res.params
pvals = res.pvalues
conf = res.conf_int()

summary = {
    "n_rows": int(n_rows),
    "num_chimps": int(num_chimps),
    "mean_efficiency": float(df["efficiency"].mean()),
    "sd_efficiency": float(df["efficiency"].std()),
    "mean_by_sex": {k: float(v) for k, v in mean_by_sex.items()},
    "mean_by_help": {k: float(v) for k, v in mean_by_help.items()},
    "age_corr": float(age_corr),
    "coef": {k: float(v) for k, v in coef.items()},
    "pvals": {k: float(v) for k, v in pvals.items()},
    "conf_int": {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in conf.index},
    "r2": float(res.rsquared),
}

print(json.dumps(summary, indent=2))
