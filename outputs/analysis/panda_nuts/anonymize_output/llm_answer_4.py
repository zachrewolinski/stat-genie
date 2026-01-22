def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels MixedLMResults (or wrapper) object
    to evaluate how Age, Sex, and HelpReceived (and their interactions) relate
    to LogEfficiency.

    Returns:
      {
        "object": dict of extracted statistics per term + model-level stats,
        "description": human-readable summary interpreting those statistics
      }
    """
    import re
    import math

    res = {}

    # Try to access core result attributes (works for MixedLMResultsWrapper)
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        ci = model_output.conf_int()  # DataFrame-like with two columns (lower, upper)
    except Exception as e:
        raise ValueError(f"Provided model_output does not expose expected attributes: {e}")

    # Helper to safely extract stats for a given parameter name
    def make_stat_entry(name):
        if name in params.index:
            return {
                "coef": float(params.loc[name]),
                "se": float(bse.loc[name]) if name in bse.index else None,
                "pvalue": float(pvalues.loc[name]) if name in pvalues.index else None,
                "ci_lower": float(ci.loc[name, 0]) if name in ci.index else None,
                "ci_upper": float(ci.loc[name, 1]) if name in ci.index else None,
                "significant_0.05": (float(pvalues.loc[name]) < 0.05) if name in pvalues.index and not math.isnan(float(pvalues.loc[name])) else None
            }
        else:
            return None

    param_names = list(params.index)

    # Target terms and patterns we want to extract
    # Exact names for numeric predictors are likely 'Age' and 'HelpReceived'
    # Sex will usually appear as a dummy like 'Sex[T.M]' or 'Sex[T.F]'
    # Interactions are typically 'Age:HelpReceived' and 'Sex[T.x]:HelpReceived'
    extracted = {}

    # Age (main)
    if "Age" in param_names:
        extracted["Age"] = make_stat_entry("Age")
    else:
        # Try to find any parameter that exactly matches 'Age' ignoring whitespace
        matches = [n for n in param_names if re.fullmatch(r".*Age.*", n)]
        extracted["Age"] = make_stat_entry(matches[0]) if matches else None

    # HelpReceived (main)
    if "HelpReceived" in param_names:
        extracted["HelpReceived"] = make_stat_entry("HelpReceived")
    else:
        matches = [n for n in param_names if re.fullmatch(r".*HelpReceived.*", n)]
        extracted["HelpReceived"] = make_stat_entry(matches[0]) if matches else None

    # Age:HelpReceived interaction(s) - there should usually be one parameter
    age_help_matches = [n for n in param_names if ("Age" in n and "HelpReceived" in n)]
    if age_help_matches:
        # If multiple (unlikely), include all
        extracted["Age:HelpReceived"] = {n: make_stat_entry(n) for n in age_help_matches}
    else:
        extracted["Age:HelpReceived"] = None

    # Sex main effect(s) (all sex-related main effect params)
    sex_main_matches = [n for n in param_names if ("Sex" in n) and (":" not in n)]
    if sex_main_matches:
        extracted["Sex_main"] = {n: make_stat_entry(n) for n in sex_main_matches}
    else:
        extracted["Sex_main"] = None

    # Sex:HelpReceived interactions (could be Sex[T.M]:HelpReceived etc.)
    sex_help_matches = [n for n in param_names if ("Sex" in n) and ("HelpReceived" in n)]
    if sex_help_matches:
        extracted["Sex:HelpReceived"] = {n: make_stat_entry(n) for n in sex_help_matches}
    else:
        extracted["Sex:HelpReceived"] = None

    # Controls: HammerType parameters (if present)
    hammer_matches = [n for n in param_names if "HammerType" in n or "C(HammerType)" in n]
    if hammer_matches:
        extracted["HammerType"] = {n: make_stat_entry(n) for n in hammer_matches}
    else:
        extracted["HammerType"] = None

    # Model-level stats: random intercept variance (if available), residual variance (scale), AIC/BIC, nobs
    model_stats = {}
    try:
        # Random-effects covariance matrix (1x1 for random intercept model)
        if hasattr(model_output, "cov_re"):
            cov_re = model_output.cov_re
            # If it's a DataFrame-like
            try:
                # Get the first diagonal element
                model_stats["random_intercept_variance"] = float(cov_re.iloc[0, 0])
            except Exception:
                # fallback: if it's an array
                model_stats["random_intercept_variance"] = float(cov_re[0][0])
        else:
            model_stats["random_intercept_variance"] = None
    except Exception:
        model_stats["random_intercept_variance"] = None

    # Residual variance (scale)
    try:
        model_stats["residual_variance"] = float(model_output.scale) if hasattr(model_output, "scale") else None
    except Exception:
        model_stats["residual_variance"] = None

    # AIC, BIC, logLik, nobs
    model_stats["AIC"] = float(model_output.aic) if hasattr(model_output, "aic") else None
    model_stats["BIC"] = float(model_output.bic) if hasattr(model_output, "bic") else None
    model_stats["logLik"] = float(model_output.llf) if hasattr(model_output, "llf") else None
    model_stats["nobs"] = int(model_output.nobs) if hasattr(model_output, "nobs") else None

    # Attach everything to the returned object
    res["terms"] = extracted
    res["model_stats"] = model_stats

    # Build a concise human-readable description summarizing key findings
    lines = []
    lines.append("Extracted fixed-effect estimates (coef, SE, p-value, 95% CI) for predictors relevant to the question.")
    # Age
    age_entry = extracted.get("Age")
    if age_entry:
        lines.append(f"- Age (main): coef={age_entry['coef']:.3f}, SE={age_entry['se']:.3f}, p={age_entry['pvalue']:.3f}, CI=[{age_entry['ci_lower']:.3f}, {age_entry['ci_upper']:.3f}]."
                     + (" Significant (p<0.05)." if age_entry["significant_0.05"] else " Not significant (p>=0.05)."))
    else:
        lines.append("- Age (main): parameter not found in model output.")

    # HelpReceived
    help_entry = extracted.get("HelpReceived")
    if help_entry:
        lines.append(f"- HelpReceived (main): coef={help_entry['coef']:.3f}, SE={help_entry['se']:.3f}, p={help_entry['pvalue']:.3f}, CI=[{help_entry['ci_lower']:.3f}, {help_entry['ci_upper']:.3f}]."
                     + (" Significant (p<0.05)." if help_entry["significant_0.05"] else " Not significant (p>=0.05)."))
    else:
        lines.append("- HelpReceived (main): parameter not found in model output.")

    # Age:HelpReceived
    if extracted.get("Age:HelpReceived"):
        for n, v in extracted["Age:HelpReceived"].items():
            if v:
                lines.append(f"- Interaction {n}: coef={v['coef']:.3f}, SE={v['se']:.3f}, p={v['pvalue']:.3f}, CI=[{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]."
                             + (" Suggests the effect of Age differs by HelpReceived (p<0.05)." if v["significant_0.05"] else " No evidence of moderation by Age (p>=0.05)."))
            else:
                lines.append(f"- Interaction {n}: stats not available.")
    else:
        lines.append("- Age:HelpReceived interaction: not present in model output.")

    # Sex main effects
    if extracted.get("Sex_main"):
        for n, v in extracted["Sex_main"].items():
            if v:
                lines.append(f"- Sex main effect ({n}): coef={v['coef']:.3f}, SE={v['se']:.3f}, p={v['pvalue']:.3f}. "
                             + ("Significant." if v["significant_0.05"] else "Not significant."))
            else:
                lines.append(f"- Sex main effect ({n}): stats not available.")
    else:
        lines.append("- Sex main effect: not present in model output.")

    # Sex:HelpReceived interactions
    if extracted.get("Sex:HelpReceived"):
        for n, v in extracted["Sex:HelpReceived"].items():
            if v:
                lines.append(f"- Interaction {n}: coef={v['coef']:.3f}, p={v['pvalue']:.3f}. "
                             + ("Indicates HelpReceived effect differs by sex (p<0.05)." if v["significant_0.05"] else "No evidence of moderation by Sex (p>=0.05)."))
            else:
                lines.append(f"- Interaction {n}: stats not available.")
    else:
        lines.append("- Sex:HelpReceived interaction: not present in model output.")

    # Model-level brief
    ri_var = model_stats.get("random_intercept_variance")
    rv = model_stats.get("residual_variance")
    lines.append(f"Model-level: random-intercept variance (ID) = {ri_var}, residual variance = {rv}, nobs = {model_stats.get('nobs')}, AIC = {model_stats.get('AIC')}, BIC = {model_stats.get('BIC')}.")

    description = " ".join(lines)

    return {"object": res, "description": description}