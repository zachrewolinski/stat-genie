import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "soccer.csv"

df = pd.read_csv(DATA_PATH)

# Ensure numeric columns
for col in ["rater1", "rater2", "redCards", "games"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute mean skin tone from available raters
skin = df[["rater1", "rater2"]].mean(axis=1)

# Keep rows with skin ratings and positive games
mask = skin.notna() & df["games"].notna() & (df["games"] > 0) & df["redCards"].notna()
sub = df.loc[mask].copy()
sub["skin_mean"] = skin[mask]

# Define light vs dark using midpoint 0.5 (exclude exactly 0.5)
sub["skin_group"] = np.where(sub["skin_mean"] > 0.5, "dark",
                              np.where(sub["skin_mean"] < 0.5, "light", "mid"))

binary = sub[sub["skin_group"].isin(["light", "dark"])].copy()

# Summary statistics
summary = (
    binary.groupby("skin_group")
    .agg(dyads=("redCards", "size"),
         total_games=("games", "sum"),
         total_red=("redCards", "sum"))
)
summary["red_per_game"] = summary["total_red"] / summary["total_games"]

# Poisson regression with log(games) offset
binary["dark"] = (binary["skin_group"] == "dark").astype(int)
X = sm.add_constant(binary["dark"])
model = sm.GLM(binary["redCards"], X, family=sm.families.Poisson(), offset=np.log(binary["games"]))
res = model.fit(cov_type="HC1")

beta = res.params["dark"]
se = res.bse["dark"]
p_value = res.pvalues["dark"]
rr = float(np.exp(beta))
ci_low = float(np.exp(beta - 1.96 * se))
ci_high = float(np.exp(beta + 1.96 * se))

# Continuous skin tone model (sensitivity)
Xc = sm.add_constant(binary["skin_mean"])
model_c = sm.GLM(binary["redCards"], Xc, family=sm.families.Poisson(), offset=np.log(binary["games"]))
res_c = model_c.fit(cov_type="HC1")

beta_c = res_c.params["skin_mean"]
se_c = res_c.bse["skin_mean"]
p_value_c = res_c.pvalues["skin_mean"]
rr_c = float(np.exp(beta_c))
ci_c_low = float(np.exp(beta_c - 1.96 * se_c))
ci_c_high = float(np.exp(beta_c + 1.96 * se_c))

output = {
    "n_rows_total": int(len(df)),
    "n_with_skin": int(len(sub)),
    "n_binary": int(len(binary)),
    "summary": summary.reset_index().to_dict(orient="records"),
    "poisson_dark_vs_light": {
        "rate_ratio": rr,
        "ci95": [ci_low, ci_high],
        "p_value": float(p_value)
    },
    "poisson_continuous": {
        "rate_ratio_per_unit": rr_c,
        "ci95": [ci_c_low, ci_c_high],
        "p_value": float(p_value_c)
    }
}

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
