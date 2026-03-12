import json
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "soccer.csv"

# Load minimal columns
cols = [
    "redCards",
    "games",
    "rater1",
    "rater2",
    "position",
    "leagueCountry",
]

# Read CSV
_df = pd.read_csv(DATA_PATH, usecols=cols)

# Compute skin tone score (mean of two raters)
_df["skin"] = _df[["rater1", "rater2"]].mean(axis=1)

# Drop rows without skin or games
_df = _df.dropna(subset=["skin", "games", "redCards"])

# Ensure games > 0 for offset
_df = _df[_df["games"] > 0]

# Basic grouped rates for light vs dark
# Define light/dark extremes on 0-1 normalized 5-point scale
light = _df[_df["skin"] <= 0.25].copy()
dark = _df[_df["skin"] >= 0.75].copy()

# Helper to compute rate

def rate(df: pd.DataFrame):
    total_red = df["redCards"].sum()
    total_games = df["games"].sum()
    return total_red / total_games if total_games > 0 else np.nan

light_rate = rate(light)
dark_rate = rate(dark)
rate_ratio = dark_rate / light_rate if (light_rate and not np.isnan(light_rate)) else np.nan

# Poisson model: dark vs light only
# Use indicator for dark (1) vs light (0) within subset
ld = pd.concat([light.assign(dark=0), dark.assign(dark=1)], axis=0)

# Guard against empty
poisson_ld = None
rr_ld = None
p_ld = None
ci_ld = None
if len(ld) > 0:
    ld = ld.copy()
    ld["log_games"] = np.log(ld["games"])
    model_ld = smf.glm(
        "redCards ~ dark", data=ld, family=sm.families.Poisson(), offset=ld["log_games"]
    )
    poisson_ld = model_ld.fit(cov_type="HC1")
    coef = poisson_ld.params.get("dark", np.nan)
    se = poisson_ld.bse.get("dark", np.nan)
    rr_ld = float(np.exp(coef)) if not np.isnan(coef) else np.nan
    # 95% CI on rate ratio
    if not np.isnan(coef) and not np.isnan(se):
        ci_low = math.exp(coef - 1.96 * se)
        ci_high = math.exp(coef + 1.96 * se)
        ci_ld = (ci_low, ci_high)
    p_ld = float(poisson_ld.pvalues.get("dark", np.nan))

# Poisson model: continuous skin tone
_df = _df.copy()
_df["log_games"] = np.log(_df["games"])

model_simple = smf.glm(
    "redCards ~ skin", data=_df, family=sm.families.Poisson(), offset=_df["log_games"]
)
res_simple = model_simple.fit(cov_type="HC1")

# With covariates for league and position
# Drop rows with missing categorical values
_df_cov = _df.dropna(subset=["position", "leagueCountry"]).copy()
model_cov = smf.glm(
    "redCards ~ skin + C(position) + C(leagueCountry)",
    data=_df_cov,
    family=sm.families.Poisson(),
    offset=_df_cov["log_games"],
)
res_cov = model_cov.fit(cov_type="HC1")

# Extract continuous effect
coef_simple = float(res_simple.params.get("skin", np.nan))
se_simple = float(res_simple.bse.get("skin", np.nan))
p_simple = float(res_simple.pvalues.get("skin", np.nan))
rr_simple = float(np.exp(coef_simple)) if not np.isnan(coef_simple) else np.nan

coef_cov = float(res_cov.params.get("skin", np.nan))
se_cov = float(res_cov.bse.get("skin", np.nan))
p_cov = float(res_cov.pvalues.get("skin", np.nan))
rr_cov = float(np.exp(coef_cov)) if not np.isnan(coef_cov) else np.nan

# Compute effect for 0.1 increase in skin tone
rr_simple_01 = float(np.exp(coef_simple * 0.1)) if not np.isnan(coef_simple) else np.nan
rr_cov_01 = float(np.exp(coef_cov * 0.1)) if not np.isnan(coef_cov) else np.nan

# Summaries for explanation
n_total = len(_df)

summary = {
    "n_total": int(n_total),
    "light_n": int(len(light)),
    "dark_n": int(len(dark)),
    "light_rate": light_rate,
    "dark_rate": dark_rate,
    "rate_ratio": rate_ratio,
    "rr_ld": rr_ld,
    "rr_ld_ci": ci_ld,
    "p_ld": p_ld,
    "rr_simple": rr_simple,
    "p_simple": p_simple,
    "rr_simple_01": rr_simple_01,
    "rr_cov": rr_cov,
    "p_cov": p_cov,
    "rr_cov_01": rr_cov_01,
}

# Decide response based on evidence
# Heuristic: if effects are positive and significant in both models, answer Yes with higher score.
# if not significant or mixed, lean No with lower score.

def pick_response(summary):
    # Default neutral
    response = 50
    direction = None

    # Determine direction using dark vs light RR
    rr = summary.get("rr_ld")
    p = summary.get("p_ld")
    if rr is not None and not np.isnan(rr):
        if rr > 1:
            direction = "positive"
        elif rr < 1:
            direction = "negative"
        else:
            direction = "neutral"

    # Significance checks
    sig_ld = p is not None and not np.isnan(p) and p < 0.05
    sig_simple = summary.get("p_simple") is not None and not np.isnan(summary.get("p_simple")) and summary.get("p_simple") < 0.05
    sig_cov = summary.get("p_cov") is not None and not np.isnan(summary.get("p_cov")) and summary.get("p_cov") < 0.05

    # Strength based on magnitude
    rr_ld = summary.get("rr_ld")
    rr_cov = summary.get("rr_cov")

    # If consistent positive significant evidence
    if direction == "positive" and sig_ld and sig_simple and sig_cov:
        # Larger RR -> higher score
        magnitude = max(rr_ld if rr_ld else 1.0, rr_cov if rr_cov else 1.0)
        if magnitude >= 1.5:
            response = 85
        elif magnitude >= 1.2:
            response = 75
        else:
            response = 65
    # If consistent negative significant evidence (dark less likely)
    elif direction == "negative" and sig_ld and sig_simple and sig_cov:
        magnitude = min(rr_ld if rr_ld else 1.0, rr_cov if rr_cov else 1.0)
        if magnitude <= 0.67:
            response = 15
        elif magnitude <= 0.83:
            response = 25
        else:
            response = 35
    else:
        # Mixed or non-significant: lean No with lower strength
        # If estimates are positive but not significant, slight lean to No
        if direction == "positive":
            response = 40
        elif direction == "negative":
            response = 60
        else:
            response = 50

    return int(response)

response = pick_response(summary)

# Build explanation
exp_lines = []
exp_lines.append(f"Analyzed {summary['n_total']} player-referee dyads with non-missing skin tone, games, and red cards.")
exp_lines.append(
    f"Light (skin ≤ 0.25) rate: {summary['light_rate']:.6f} red cards per game (n={summary['light_n']}); "
    f"Dark (skin ≥ 0.75) rate: {summary['dark_rate']:.6f} per game (n={summary['dark_n']})."
)
if summary["rr_ld"] is not None and not np.isnan(summary["rr_ld"]):
    if summary["rr_ld_ci"] is not None:
        ci_low, ci_high = summary["rr_ld_ci"]
        exp_lines.append(
            f"Poisson model on light vs dark with games offset: rate ratio={summary['rr_ld']:.3f} "
            f"(95% CI {ci_low:.3f}–{ci_high:.3f}), p={summary['p_ld']:.4g}."
        )
    else:
        exp_lines.append(
            f"Poisson model on light vs dark with games offset: rate ratio={summary['rr_ld']:.3f}, p={summary['p_ld']:.4g}."
        )

exp_lines.append(
    f"Continuous skin tone Poisson model (offset by games): RR for full 0–1 scale={summary['rr_simple']:.3f} "
    f"(p={summary['p_simple']:.4g}); per 0.1 increase RR={summary['rr_simple_01']:.3f}."
)
exp_lines.append(
    f"With league and position controls: RR for full 0–1 scale={summary['rr_cov']:.3f} "
    f"(p={summary['p_cov']:.4g}); per 0.1 increase RR={summary['rr_cov_01']:.3f}."
)

# Decision rationale
if response >= 60:
    decision = "Yes"
elif response <= 40:
    decision = "No"
else:
    decision = "Unclear/weak evidence"

exp_lines.append(
    f"Conclusion: {decision}. The Likert response reflects the direction and statistical significance of the estimated rate ratios across models."
)

explanation = " ".join(exp_lines)

out = {"response": response, "explanation": explanation}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(out, f)
