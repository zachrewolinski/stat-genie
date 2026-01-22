def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for the predictors
    of interest (Age, Sex_F, Help) from a statsmodels MixedLMResults / wrapper object.
    Returns a dict with keys:
      - "object": dictionary mapping variable -> stats dictionary
      - "description": human-readable summary interpreting each effect
    """
    import numpy as np

    # Helper to safely access attributes with fallbacks
    def _get_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Try to extract core result tables
    params = _get_attr(model_output, "params", None)
    bse = _get_attr(model_output, "bse", None)
    pvalues = _get_attr(model_output, "pvalues", None)
    tvalues = _get_attr(model_output, "tvalues", None)
    try:
        conf_int_df = model_output.conf_int()
    except Exception:
        conf_int_df = None

    # If p-values are missing but t-values exist, approximate p-values using normal dist
    if pvalues is None and tvalues is not None:
        try:
            from scipy import stats
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))
        except Exception:
            # fallback: mark p-values as unavailable
            pvalues = None

    predictors = ["Age", "Sex_F", "Help"]
    result_obj = {}
    summary_lines = []

    for pred in predictors:
        if params is None or pred not in params.index:
            result_obj[pred] = None
            summary_lines.append(f"{pred}: not present in model results.")
            continue

        coef = float(params.loc[pred])
        se = float(bse.loc[pred]) if (bse is not None and pred in bse.index) else None
        pval = float(pvalues.loc[pred]) if (pvalues is not None and pred in pvalues.index) else None

        if conf_int_df is not None and pred in conf_int_df.index:
            ci_lower = float(conf_int_df.loc[pred, 0])
            ci_upper = float(conf_int_df.loc[pred, 1])
        else:
            # approximate 95% CI if se available
            if se is not None:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = ci_upper = None

        # Interpret coefficient on log scale: percent change = (exp(coef)-1)*100
        try:
            pct_change = (np.exp(coef) - 1) * 100.0
        except Exception:
            pct_change = None

        sig = None
        if pval is not None:
            sig = (pval < 0.05)

        result_obj[pred] = {
            "coef_log_scale": coef,
            "std_error": se,
            "p_value": pval,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "percent_change_per_unit": pct_change,  # multiplicative change in efficiency per 1 unit (in %)
            "significant_at_0.05": sig,
        }

        # Build a human-readable line
        line = f"{pred}: coef={coef:.4f}"
        if se is not None:
            line += f", SE={se:.4f}"
        if pval is not None:
            line += f", p={pval:.3f}"
        if (ci_lower is not None) and (ci_upper is not None):
            line += f", 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]"
        if pct_change is not None:
            line += f" -> ≈{pct_change:.1f}% change in efficiency per unit"
        if sig is True:
            line += " (statistically significant at α=0.05)"
        elif sig is False:
            line += " (not statistically significant at α=0.05)"
        summary_lines.append(line)

    # Compose overall description
    description = (
        "Extracted model estimates for key predictors affecting logged nut-cracking efficiency.\n"
        + "\n".join(summary_lines)
        + "\n\nNotes:\n"
        "- Coefficients are on the log(nuts opened per second) scale. Positive coef => higher efficiency.\n"
        "- For Sex_F (female=1, male=0): a positive coef means females are more efficient than males.\n"
        "- For Help (1=yes): a positive coef means sessions with help are more efficient.\n"
        "- Percent change = (exp(coef)-1)*100 gives approximate multiplicative change in raw efficiency per one-unit increase.\n"
        "- If any predictor is missing above, it was not present in the fitted model's parameter table."
    )

    return {"object": result_obj, "description": description}