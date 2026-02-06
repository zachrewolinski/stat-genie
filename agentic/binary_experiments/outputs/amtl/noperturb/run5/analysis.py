import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Basic cleaning
_df = _df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]).copy()
_df = _df[_df["sockets"] > 0].copy()
_df = _df[_df["num_amtl"] <= _df["sockets"]].copy()

_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)
_df["rate"] = _df["num_amtl"] / _df["sockets"]

# Binomial regression with trials as weights
formula = "rate ~ is_human + age + prob_male + C(tooth_class)"
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df["sockets"],
).fit()

# Extract effect for human vs non-human
coef = model.params.get("is_human", np.nan)
se = model.bse.get("is_human", np.nan)
pval = model.pvalues.get("is_human", np.nan)

# Odds ratio and 95% CI
or_val = float(np.exp(coef))
ci_low = float(np.exp(coef - 1.96 * se))
ci_high = float(np.exp(coef + 1.96 * se))

# Predicted marginal rates (weighted by sockets)
_df_non = _df.copy()
_df_non["is_human"] = 0
_df_hum = _df.copy()
_df_hum["is_human"] = 1

pred_non = model.predict(_df_non)
pred_hum = model.predict(_df_hum)
weights = _df["sockets"].to_numpy()

marg_non = float(np.average(pred_non, weights=weights))
marg_hum = float(np.average(pred_hum, weights=weights))

# Descriptive rates by genus (weighted by sockets)
_desc = (
    _df.assign(weight=_df["sockets"])
      .groupby("genus")
      .apply(lambda g: np.average(g["rate"], weights=g["weight"]))
      .sort_values(ascending=False)
)

print("Model summary (truncated):")
print(model.summary().tables[1])
print("\nEffect of human vs non-human (log-odds):")
print(f"coef={coef:.4f}, se={se:.4f}, p={pval:.4g}")
print(f"odds_ratio={or_val:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")
print("\nPredicted marginal AMTL rate:")
print(f"human={marg_hum:.4f}, non-human={marg_non:.4f}, diff={marg_hum-marg_non:.4f}")
print("\nWeighted mean AMTL rate by genus:")
print(_desc)

# Save key results for use in conclusion
with open("analysis_results.txt", "w") as f:
    f.write(f"coef_is_human={coef}\n")
    f.write(f"pval_is_human={pval}\n")
    f.write(f"or_is_human={or_val}\n")
    f.write(f"ci_low={ci_low}\n")
    f.write(f"ci_high={ci_high}\n")
    f.write(f"marg_human={marg_hum}\n")
    f.write(f"marg_nonhuman={marg_non}\n")
    f.write(f"diff={marg_hum-marg_non}\n")
