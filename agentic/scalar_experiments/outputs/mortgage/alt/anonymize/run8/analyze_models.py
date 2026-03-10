import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("mortgage.csv")

outcome = "feature14"

def fit_model(cols, label):
    data = df[[outcome] + cols].dropna()
    y = data[outcome]
    X = sm.add_constant(data[cols], has_constant="add")
    res = sm.Logit(y, X).fit(disp=False)
    coef = res.params.get("feature2", np.nan)
    pval = res.pvalues.get("feature2", np.nan)
    or_adj = np.exp(coef) if pd.notnull(coef) else np.nan
    print(label)
    print("n=", len(data), "coef=", coef, "p=", pval, "OR=", or_adj)

# Model A: gender only
fit_model(["feature2"], "Model A: gender only")

# Model B: full controls (exclude feature1,11,14)
full_cols = [c for c in df.columns if c not in ["feature1", "feature11", "feature14"]]
fit_model(full_cols, "Model B: full controls")

# Model C: full controls except race
cols_no_race = [c for c in full_cols if c != "feature3"]
fit_model(cols_no_race, "Model C: full controls except race")

# Model D: credit-focused controls
credit_cols = ["feature2", "feature7", "feature8", "feature9", "feature10", "feature12", "feature4"]
fit_model(credit_cols, "Model D: credit-focused")
