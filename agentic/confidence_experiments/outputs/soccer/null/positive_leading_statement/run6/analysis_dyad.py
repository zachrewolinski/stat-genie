import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
path = "soccer.csv"
df = pd.read_csv(path)

# Skin tone average
skin = df[["rater1", "rater2"]].mean(axis=1)
df = df.assign(skin_tone=skin)

# Drop missing
df_use = df[["playerShort", "games", "redCards", "skin_tone"]].dropna()

# dyad-level rate
rate = df_use["redCards"] / df_use["games"]

# Spearman correlation
spearman = stats.spearmanr(df_use["skin_tone"], rate, nan_policy="omit")

# Poisson regression with offset
X = sm.add_constant(df_use["skin_tone"])
offset = np.log(df_use["games"])
poisson_model = sm.GLM(df_use["redCards"], X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit(cov_type='cluster', cov_kwds={'groups': df_use['playerShort']})

result = {
    "n_dyads": int(df_use.shape[0]),
    "poisson_skin": {
        "coef": float(poisson_res.params["skin_tone"]),
        "se": float(poisson_res.bse["skin_tone"]),
        "pvalue": float(poisson_res.pvalues["skin_tone"]),
        "exp_coef": float(np.exp(poisson_res.params["skin_tone"])),
    },
    "spearman": {
        "rho": float(spearman.correlation),
        "pvalue": float(spearman.pvalue),
    },
}

print(json.dumps(result, indent=2))
