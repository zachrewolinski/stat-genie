import pandas as pd
import numpy as np
import statsmodels.api as sm
import json

# Load data
csv_path = "soccer.csv"

df = pd.read_csv(csv_path)

# Basic cleaning
# Use mean of rater1 and rater2 for skin tone when both available
skin = df[["rater1", "rater2"]].mean(axis=1)

df = df.copy()

df["skin_avg"] = skin

# Define dark vs light: exclude exactly 0.5 (neutral) to compare dark vs light
# dark: > 0.5, light: < 0.5
# Use only rows with skin ratings
mask_skin = df["skin_avg"].notna()

df_skin = df[mask_skin].copy()

# Calculate rates per game
# Avoid division by zero by filtering games > 0

df_skin = df_skin[df_skin["games"] > 0]

# Define groups

df_skin["skin_group"] = np.where(df_skin["skin_avg"] > 0.5, "dark",
                                 np.where(df_skin["skin_avg"] < 0.5, "light", "mid"))

# Only dark vs light

df_dl = df_skin[df_skin["skin_group"].isin(["dark", "light"])].copy()

# Compute summary stats

def group_summary(sub):
    total_games = sub["games"].sum()
    total_reds = sub["redCards"].sum()
    rate = total_reds / total_games if total_games > 0 else np.nan
    return pd.Series({
        "n_dyads": len(sub),
        "total_games": total_games,
        "total_red_cards": total_reds,
        "red_cards_per_game": rate,
        "mean_redCards": sub["redCards"].mean(),
        "mean_games": sub["games"].mean(),
    })

summary = df_dl.groupby("skin_group").apply(group_summary)

# Rate ratio (dark/light)
if "dark" in summary.index and "light" in summary.index:
    rate_ratio = summary.loc["dark", "red_cards_per_game"] / summary.loc["light", "red_cards_per_game"]
else:
    rate_ratio = np.nan

# Poisson regression with offset log(games)
# Model 1: continuous skin_avg

model_data = df_skin.copy()
model_data = model_data[model_data["games"] > 0]
model_data = model_data[model_data["skin_avg"].notna()]

# Add intercept

model_data = model_data.assign(log_games=np.log(model_data["games"]))

X = sm.add_constant(model_data["skin_avg"])

poisson_model = sm.GLM(model_data["redCards"], X, family=sm.families.Poisson(), offset=model_data["log_games"])

poisson_res = poisson_model.fit()

# Robust standard errors clustered by playerShort (if available)

robust_res = None
if "playerShort" in model_data.columns:
    try:
        robust_res = poisson_res.get_robustcov_results(cov_type='cluster', groups=model_data["playerShort"])
    except Exception:
        robust_res = None

# Poisson regression with dark indicator

model_dl = df_dl.copy()
model_dl = model_dl.assign(dark=(model_dl["skin_group"] == "dark").astype(int))
model_dl = model_dl.assign(log_games=np.log(model_dl["games"]))
X_dl = sm.add_constant(model_dl["dark"])

poisson_dl = sm.GLM(model_dl["redCards"], X_dl, family=sm.families.Poisson(), offset=model_dl["log_games"])
poisson_dl_res = poisson_dl.fit()

robust_dl_res = None
if "playerShort" in model_dl.columns:
    try:
        robust_dl_res = poisson_dl_res.get_robustcov_results(cov_type='cluster', groups=model_dl["playerShort"])
    except Exception:
        robust_dl_res = None

# Extract results

def extract_result(res):
    coef = res.params[1]  # skin_avg or dark
    se = res.bse[1]
    p = res.pvalues[1]
    rr = np.exp(coef)
    return coef, se, p, rr

coef_cont, se_cont, p_cont, rr_cont = extract_result(robust_res or poisson_res)
coef_dark, se_dark, p_dark, rr_dark = extract_result(robust_dl_res or poisson_dl_res)

out = {
    "summary": summary.to_dict(),
    "rate_ratio_dark_light": rate_ratio,
    "poisson_continuous": {
        "coef": coef_cont,
        "se": se_cont,
        "p_value": p_cont,
        "rate_ratio_per_unit_skin": rr_cont,
    },
    "poisson_dark_indicator": {
        "coef": coef_dark,
        "se": se_dark,
        "p_value": p_dark,
        "rate_ratio_dark_vs_light": rr_dark,
    },
    "n_total": len(df_skin),
    "n_dark_light": len(df_dl),
}

with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
