def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression).
    Returns a dict with:
      - "object": dict of per-predictor statistics (coef, se, z, p, 95% CI, odds ratio and OR 95% CI, significance)
      - "description": short interpretation of the main predictors in the context of the task
    
    Expects model_output to have attributes: params, bse, pvalues, conf_int().
    """
    import numpy as np
    import pandas as pd

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    params = model_output.params.copy()
    bse = model_output.bse.copy()
    pvalues = model_output.pvalues.copy()
    # confidence intervals: statsmodels method .conf_int()
    try:
        ci = model_output.conf_int()
    except Exception as e:
        # If conf_int not available, set NaNs
        ci = np.full((len(params), 2), np.nan)

    # Ensure we have an index for confidence intervals
    try:
        ci_df = pd.DataFrame(ci, index=params.index, columns=["2.5%", "97.5%"])
    except Exception:
        ci_df = pd.DataFrame(ci, columns=["2.5%", "97.5%"])
        # if no index, align by position with params index
        ci_df.index = params.index

    # z (or t) statistic
    with np.errstate(divide="ignore", invalid="ignore"):
        zvals = params / bse

    # Odds ratios and CI on OR scale
    or_vals = np.exp(params)
    or_ci_low = np.exp(ci_df["2.5%"].astype(float))
    or_ci_high = np.exp(ci_df["97.5%"].astype(float))

    # Build results per predictor
    predictors = {}
    for name in params.index:
        predictors[name] = {
            "coef": float(params[name]) if not pd.isna(params[name]) else None,
            "se": float(bse[name]) if name in bse.index and not pd.isna(bse[name]) else None,
            "z": float(zvals[name]) if name in params.index and not pd.isna(zvals[name]) else None,
            "p_value": float(pvalues[name]) if name in pvalues.index and not pd.isna(pvalues[name]) else None,
            "ci_2.5%": float(ci_df.loc[name, "2.5%"]) if name in ci_df.index and not pd.isna(ci_df.loc[name, "2.5%"]) else None,
            "ci_97.5%": float(ci_df.loc[name, "97.5%"]) if name in ci_df.index and not pd.isna(ci_df.loc[name, "97.5%"]) else None,
            "odds_ratio": float(or_vals[name]) if not pd.isna(or_vals[name]) else None,
            "odds_ratio_CI_low": float(or_ci_low[name]) if name in or_ci_low.index and not pd.isna(or_ci_low[name]) else None,
            "odds_ratio_CI_high": float(or_ci_high[name]) if name in or_ci_high.index and not pd.isna(or_ci_high[name]) else None,
            "significant_p_lt_0.05": bool((name in pvalues.index) and (not pd.isna(pvalues[name])) and (pvalues[name] < 0.05))
        }

    # Compose a short interpretation focused on the task variables:
    key_terms = ["RelGroupSize_c", "RelLocation_c", "RelGroupSize_c:RelLocation_c"]
    interp_lines = []
    for term in key_terms:
        if term in predictors:
            info = predictors[term]
            p = info["p_value"]
            coef = info["coef"]
            orv = info["odds_ratio"]
            sig = info["significant_p_lt_0.05"]
            if p is None:
                interp_lines.append(f"{term}: no estimate available.")
            else:
                sign_text = "positive" if coef > 0 else ("negative" if coef < 0 else "near zero")
                sig_text = "statistically significant (p < 0.05)" if sig else f"not statistically significant (p = {p:.3g})"
                interp_lines.append(
                    f"{term}: coef = {coef:.3f}, OR = {orv:.3f}, {sig_text}; sign: {sign_text}."
                )
        else:
            interp_lines.append(f"{term}: not present in model results.")

    description = (
        "Extracted coefficients, SEs, z-values, p-values, 95% CIs and odds ratios for model predictors. "
        "For the task, focus on RelGroupSize_c (relative group size advantage), RelLocation_c (relative location), "
        "and their interaction. Positive coefficients mean higher log-odds (and OR>1 means higher odds) of the focal group winning. "
        "Summary for focal predictors: " + " ".join(interp_lines)
    )

    return {"object": predictors, "description": description}