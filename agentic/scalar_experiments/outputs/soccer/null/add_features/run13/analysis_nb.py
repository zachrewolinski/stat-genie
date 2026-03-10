import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

for col in ["rater1", "rater2", "redCards", "games"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
clean = df.assign(skin_mean=skin).dropna(subset=["skin_mean", "games", "redCards"]).copy()
clean = clean[clean["games"] > 0]
clean["dark"] = (clean["skin_mean"] > 0.5).astype(int)

# Negative binomial with log(games) offset (dark indicator)
X = sm.add_constant(clean["dark"])
offset = np.log(clean["games"])
nb = smd.NegativeBinomial(clean["redCards"], X, offset=offset).fit(disp=False)

coef = nb.params["dark"]
se = nb.bse["dark"]
ci = nb.conf_int().loc["dark"].tolist()
pval = nb.pvalues["dark"]

irr = np.exp(coef)
ci_irr = np.exp(ci)

print("Negative Binomial (dark): coef=%.4f, SE=%.4f, p=%.4g" % (coef, se, pval))
print("IRR=%.4f, 95%% CI=[%.4f, %.4f]" % (irr, ci_irr[0], ci_irr[1]))
print("alpha (overdispersion) =", nb.params.get("alpha"))
