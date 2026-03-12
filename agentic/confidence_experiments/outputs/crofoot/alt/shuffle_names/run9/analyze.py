import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Map columns based on info.json descriptions
# Outcome: m_focal (1 if focal wins)
# Relative group size: f_other (focal size) vs win (other size)
# Contest location: m_other (focal distance to its home range center) vs n_focal (other distance)

outcome = df["m_focal"].astype(int)
rel_size = df["f_other"] - df["win"]
rel_dist = df["m_other"] - df["n_focal"]

# Build design matrix with intercept
X = pd.DataFrame({
    "rel_size": rel_size,
    "rel_dist": rel_dist,
})
X = sm.add_constant(X)

model = sm.Logit(outcome, X).fit(disp=False)

# Also compute standardized coefficients for interpretation
X_std = X.copy()
for col in ["rel_size", "rel_dist"]:
    X_std[col] = (X_std[col] - X_std[col].mean()) / X_std[col].std(ddof=0)
model_std = sm.Logit(outcome, X_std).fit(disp=False)

conf_int = model.conf_int()
results = {
    "n": int(len(df)),
    "coef": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "coef_std": model_std.params.to_dict(),
    "pvalues_std": model_std.pvalues.to_dict(),
    "conf_int": {idx: [float(conf_int.loc[idx, 0]), float(conf_int.loc[idx, 1])] for idx in conf_int.index},
}

print(json.dumps(results, indent=2))
