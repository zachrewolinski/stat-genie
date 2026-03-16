import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = Path(__file__).with_name("panda_nuts.csv")
OUTPUT_PATH = Path(__file__).with_name("conclusion.txt")


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.copy()
    for col in ["sex", "help"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Standardize help to y/n when possible
    df.loc[df["help"].isin(["n", "no", "0", "false", "f"]), "help"] = "n"
    df.loc[df["help"].isin(["y", "yes", "1", "true", "t"]), "help"] = "y"

    # Efficiency: nuts per second
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    # Drop rows with missing values in key fields
    df = df.dropna(subset=["efficiency", "age", "sex", "help", "chimpanzee"])

    n_rows = len(df)
    n_chimps = df["chimpanzee"].nunique()

    # OLS with cluster-robust SE by chimpanzee (accounts for repeated sessions)
    ols_model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df)
    try:
        ols_res = ols_model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})
        ols_ok = True
    except Exception:
        ols_res = ols_model.fit()
        ols_ok = False

    # Try mixed effects model with random intercept per chimpanzee
    mixed_res = None
    mixed_ok = False
    try:
        mixed_model = smf.mixedlm("efficiency ~ age + C(sex) + C(help)", data=df, groups=df["chimpanzee"])
        mixed_res = mixed_model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        mixed_ok = True
    except Exception:
        mixed_ok = False

    # Helper to extract p-values from a fitted model
    def get_pvals(res):
        p = res.pvalues.copy()
        # Ensure consistent keys even if reference categories vary
        return {
            "age": float(p.get("age", np.nan)),
            "sex": float(p.get("C(sex)[T.m]", np.nan)),
            "help": float(p.get("C(help)[T.y]", np.nan)),
        }

    pvals_ols = get_pvals(ols_res)
    pvals_mixed = get_pvals(mixed_res) if mixed_ok else None

    # Effect sizes (model coefficients)
    def get_effects(res):
        params = res.params
        return {
            "age": float(params.get("age", np.nan)),
            "sex": float(params.get("C(sex)[T.m]", np.nan)),
            "help": float(params.get("C(help)[T.y]", np.nan)),
        }

    effects_ols = get_effects(ols_res)
    effects_mixed = get_effects(mixed_res) if mixed_ok else None

    # Group means for context
    mean_eff = df["efficiency"].mean()
    mean_by_sex = df.groupby("sex")["efficiency"].mean().to_dict()
    mean_by_help = df.groupby("help")["efficiency"].mean().to_dict()

    # Simple Welch t-tests (ignoring clustering) for descriptive support
    def welch_p(col, a, b):
        xa = df.loc[df[col] == a, "efficiency"]
        xb = df.loc[df[col] == b, "efficiency"]
        if len(xa) < 2 or len(xb) < 2:
            return math.nan
        return float(stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit").pvalue)

    sex_p = welch_p("sex", "m", "f")
    help_p = welch_p("help", "y", "n")

    # Determine evidence: consider both mixed and clustered OLS when available
    def is_sig(p):
        return p == p and p < 0.05

    def combine_sig(ols_p, mixed_p):
        if is_sig(ols_p):
            return True
        if mixed_p is None:
            return False
        return is_sig(mixed_p)

    sig_age = combine_sig(pvals_ols["age"], pvals_mixed["age"] if pvals_mixed else None)
    sig_sex = combine_sig(pvals_ols["sex"], pvals_mixed["sex"] if pvals_mixed else None)
    sig_help = combine_sig(pvals_ols["help"], pvals_mixed["help"] if pvals_mixed else None)

    # Score heuristic based on strength of evidence
    sig_count = sum([sig_age, sig_sex, sig_help])
    if sig_count == 0:
        response = 20
    elif sig_count == 1:
        response = 55
    elif sig_count == 2:
        response = 70
    else:
        response = 85

    # Adjust slightly based on effect sizes and consistency between models
    if mixed_ok:
        # if mixed and clustered OLS disagree on significance for any variable, dampen confidence
        disagree = 0
        for key, flag in [("age", sig_age), ("sex", sig_sex), ("help", sig_help)]:
            ols_sig = is_sig(pvals_ols[key])
            mixed_sig = is_sig(pvals_mixed[key])
            if ols_sig != mixed_sig:
                disagree += 1
        if disagree >= 2:
            response = max(0, response - 10)
        elif disagree == 1:
            response = max(0, response - 5)

    response = int(round(min(100, max(0, response))))

    # Build explanation
    lines = []
    lines.append(f"Data: {n_rows} sessions from {n_chimps} chimpanzees. Efficiency defined as nuts_opened/seconds (nuts per second).")

    # OLS summary
    lines.append(
        "Cluster-robust OLS (clustered by chimpanzee) fixed effects: "
        f"age coef={effects_ols['age']:.4f}, p={pvals_ols['age']:.4g}; "
        f"sex(male vs female) coef={effects_ols['sex']:.4f}, p={pvals_ols['sex']:.4g}; "
        f"help(yes vs no) coef={effects_ols['help']:.4f}, p={pvals_ols['help']:.4g}."
    )

    if mixed_ok:
        lines.append(
            "Mixed-effects model with random intercept by chimpanzee: "
            f"age coef={effects_mixed['age']:.4f}, p={pvals_mixed['age']:.4g}; "
            f"sex(male vs female) coef={effects_mixed['sex']:.4f}, p={pvals_mixed['sex']:.4g}; "
            f"help(yes vs no) coef={effects_mixed['help']:.4f}, p={pvals_mixed['help']:.4g}."
        )
    else:
        lines.append("Mixed-effects model did not converge; relying on clustered OLS for inference.")

    lines.append(
        f"Mean efficiency overall={mean_eff:.4f}; by sex={mean_by_sex}; by help={mean_by_help}. "
        f"Welch t-tests (descriptive, no clustering): sex p={sex_p:.4g}, help p={help_p:.4g}."
    )

    # Interpret
    sig_labels = []
    if sig_age:
        sig_labels.append("age")
    if sig_sex:
        sig_labels.append("sex")
    if sig_help:
        sig_labels.append("help")

    if sig_labels:
        lines.append(
            "Evidence of influence is strongest for: " + ", ".join(sig_labels) +
            ". Variables not listed show weak or non-significant effects at alpha=0.05."
        )
    else:
        lines.append("No predictor shows a statistically significant effect at alpha=0.05 in the clustered model.")

    explanation = " ".join(lines)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"response": response, "explanation": explanation}, f, ensure_ascii=True)

    # Also print a brief summary for interactive inspection
    print(json.dumps({
        "response": response,
        "sig_age": sig_age,
        "sig_sex": sig_sex,
        "sig_help": sig_help,
        "pvals_ols": pvals_ols,
        "pvals_mixed": pvals_mixed,
    }, indent=2))


if __name__ == "__main__":
    main()
