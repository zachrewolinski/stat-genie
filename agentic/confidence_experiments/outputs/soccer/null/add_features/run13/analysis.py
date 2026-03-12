import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
csv_path = "soccer.csv"
df = pd.read_csv(csv_path)

# Ensure numeric columns
for col in ["rater1", "rater2", "redCards", "games"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Skin tone: mean of available raters
skin = df[["rater1", "rater2"]].mean(axis=1, skipna=True)
df = df.assign(skin_mean=skin)

# Keep rows with skin rating and valid games/redCards
clean = df.dropna(subset=["skin_mean", "games", "redCards"]).copy()
clean = clean[clean["games"] > 0]

# Dark vs light split (dark > 0.5 on normalized 0-1 scale)
clean["dark"] = (clean["skin_mean"] > 0.5).astype(int)

# Group summaries
summary = (
    clean.groupby("dark")
    .agg(dyads=("dark", "size"), games=("games", "sum"), redCards=("redCards", "sum"))
    .assign(rate_per_game=lambda x: x["redCards"] / x["games"])
)

# Poisson regression with offset (continuous skin)
X_cont = sm.add_constant(clean["skin_mean"])
model_cont = sm.GLM(
    clean["redCards"], X_cont, family=sm.families.Poisson(), offset=np.log(clean["games"])
).fit(cov_type="HC1")

# Poisson regression with offset (dark indicator)
X_bin = sm.add_constant(clean["dark"])
model_bin = sm.GLM(
    clean["redCards"], X_bin, family=sm.families.Poisson(), offset=np.log(clean["games"])
).fit(cov_type="HC1")

# Overdispersion check (Pearson chi2 / df_resid)
pearson_chi2 = ((clean["redCards"] - model_cont.mu) ** 2 / model_cont.mu).sum()
overdisp = pearson_chi2 / model_cont.df_resid

# Extract stats
coef_cont = model_cont.params["skin_mean"]
se_cont = model_cont.bse["skin_mean"]
pp_cont = model_cont.pvalues["skin_mean"]
ci_cont = model_cont.conf_int().loc["skin_mean"].tolist()

coef_bin = model_bin.params["dark"]
se_bin = model_bin.bse["dark"]
pp_bin = model_bin.pvalues["dark"]
ci_bin = model_bin.conf_int().loc["dark"].tolist()

# Transform to incidence rate ratios
irr_cont = np.exp(coef_cont)
irr_cont_ci = np.exp(ci_cont)

irr_bin = np.exp(coef_bin)
irr_bin_ci = np.exp(ci_bin)

# Print results
print("Rows with skin rating:", len(clean))
print("Summary by dark group (0=light/medium, 1=dark):")
print(summary)
print("\nPoisson (continuous skin): coef=%.4f, SE=%.4f, p=%.4g" % (coef_cont, se_cont, pp_cont))
print("IRR=%.4f, 95%% CI=[%.4f, %.4f]" % (irr_cont, irr_cont_ci[0], irr_cont_ci[1]))
print("\nPoisson (dark indicator): coef=%.4f, SE=%.4f, p=%.4g" % (coef_bin, se_bin, pp_bin))
print("IRR=%.4f, 95%% CI=[%.4f, %.4f]" % (irr_bin, irr_bin_ci[0], irr_bin_ci[1]))
print("\nOverdispersion (Pearson chi2/df): %.3f" % overdisp)
