def extract_final_answer(model_output):
    """
    Extract statistics about the 'female' coefficient from a fitted statsmodels
    binary regression results object (Logit/GLM or their wrapper).

    Returns a dict with:
      - "object": a dict of numeric results for 'female' (coef, se, z, p, conf_int,
                  odds_ratio, odds_ratio_conf_int, (if available) average marginal effect)
      - "description": short interpretation in plain language.

    The function is defensive: it converts numpy types to native Python floats and
    falls back gracefully if some summaries (e.g., marginal effects) are not available.
    """
    import numpy as np

    res = model_output

    # Helper to safely access attributes
    def safe_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Try to get parameter series / dict-like
    try:
        params = safe_attr(res, "params", None)
        if params is None:
            # For some wrapper objects params might be accessed slightly differently
            params = res.params
    except Exception:
        raise ValueError("Could not extract params from model_output")

    if "female" not in params.index:
        raise ValueError("Model output does not contain a 'female' coefficient")

    # Extract coefficient and standard error
    coef = float(params["female"])

    # standard error
    try:
        bse = float(safe_attr(res, "bse")[ "female" ])
    except Exception:
        # try alternative attribute names
        try:
            bse = float(res.bse["female"])
        except Exception:
            bse = None

    # p-value
    try:
        pval = float(safe_attr(res, "pvalues")["female"])
    except Exception:
        try:
            pval = float(res.pvalues["female"])
        except Exception:
            pval = None

    # z/t statistic: compute as coef / se if possible
    z_stat = None
    if bse is not None and bse != 0:
        z_stat = float(coef / bse)

    # confidence interval
    ci_low = ci_high = None
    try:
        ci = safe_attr(res, "conf_int")()
        # conf_int() returns array like [[low1, high1], ...] indexed in same order as params
        # but sometimes conf_int is an attribute (DataFrame) instead of callable
    except Exception:
        try:
            ci = safe_attr(res, "conf_int")
        except Exception:
            ci = None

    if ci is not None:
        try:
            # If ci is a DataFrame or ndarray with index matching params
            if hasattr(ci, "loc"):
                ci_low = float(ci.loc["female"].iloc[0])
                ci_high = float(ci.loc["female"].iloc[1])
            else:
                # ci might be ndarray; find the position of 'female'
                idx = list(params.index).index("female")
                ci_low = float(ci[idx, 0])
                ci_high = float(ci[idx, 1])
        except Exception:
            ci_low = ci_high = None

    # Odds ratio and its CI (exp of coef/conf_int)
    try:
        odds_ratio = float(np.exp(coef))
    except Exception:
        odds_ratio = None

    or_ci_low = or_ci_high = None
    if ci_low is not None and ci_high is not None:
        try:
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
        except Exception:
            or_ci_low = or_ci_high = None

    # Try to compute average marginal effect for 'female' (if supported)
    me_value = me_se = me_p = me_ci_low = me_ci_high = None
    try:
        # get_margeff() exists for discrete models in statsmodels
        margeff = res.get_margeff()
        # summary_frame() returns a DataFrame with index matching variables
        try:
            me_df = margeff.summary_frame()
            if "female" in me_df.index:
                me_value = float(me_df.loc["female"]["dy/dx"])
                # Standard error column name varies; try common ones
                se_col = None
                for c in ["Std. Err.", "std err", "std_err", "Std Err"]:
                    if c in me_df.columns:
                        se_col = c
                        break
                if se_col is not None:
                    me_se = float(me_df.loc["female"][se_col])
                # p-value column:
                for c in ["P>|z|", "p-value", "pvalue", "P>|z| "]:
                    if c in me_df.columns:
                        me_p = float(me_df.loc["female"][c])
                        break
                # CI columns:
                # common columns are ['[0.025', '0.975]'] or 'CI_lower','CI_upper'
                if "[0.025" in me_df.columns and "0.975]" in me_df.columns:
                    me_ci_low = float(me_df.loc["female"]["[0.025"])
                    me_ci_high = float(me_df.loc["female"]["0.975]"])
                else:
                    for lowc, highc in [("CI_lower", "CI_upper"), ("0.025", "0.975")]:
                        if lowc in me_df.columns and highc in me_df.columns:
                            me_ci_low = float(me_df.loc["female"][lowc])
                            me_ci_high = float(me_df.loc["female"][highc])
                            break
        except Exception:
            # If summary_frame fails, try to index margins directly
            try:
                me_res = margeff.summary()
                # parsing text summary is brittle; skip if not straightforward
            except Exception:
                pass
    except Exception:
        # marginal effects not available for this result object
        pass

    # Build the returned object (numeric values)
    numeric_result = {
        "coef_log_odds": coef,
        "std_err": bse,
        "z_stat": z_stat,
        "p_value": pval,
        "conf_int_95": [ci_low, ci_high],
        "odds_ratio": odds_ratio,
        "odds_ratio_conf_int_95": [or_ci_low, or_ci_high],
        "avg_marginal_effect": me_value,
        "avg_marginal_effect_se": me_se,
        "avg_marginal_effect_pvalue": me_p,
        "avg_marginal_effect_conf_int_95": [me_ci_low, me_ci_high],
    }

    # Interpretation: succinct plain-language description
    # Determine direction and significance if p-value known
    if pval is not None:
        sig = pval < 0.05
    else:
        sig = None

    if coef > 0:
        direction = "higher"
    elif coef < 0:
        direction = "lower"
    else:
        direction = "no difference"

    desc_parts = []
    desc_parts.append(
        "The model coefficient for 'female' is {:.4g} (log-odds).".format(coef)
    )
    if odds_ratio is not None:
        desc_parts.append(
            "This corresponds to an odds ratio of {:.4g}.".format(odds_ratio)
        )
        if or_ci_low is not None and or_ci_high is not None:
            desc_parts.append(
                "95% CI for the odds ratio: [{:.4g}, {:.4g}].".format(or_ci_low, or_ci_high)
            )
    if pval is not None:
        desc_parts.append(
            "p-value = {:.4g}.".format(pval) + (" (statistically significant at alpha=0.05)" if sig else " (not statistically significant at alpha=0.05)")
        )
    else:
        desc_parts.append("p-value not available from the model output.")

    # Short plain conclusion about gender effect
    if sig is True:
        desc_parts.append("Conclusion: Female applicants have {} odds of mortgage acceptance compared to male applicants, conditional on controls (statistically significant).".format(direction))
    elif sig is False:
        desc_parts.append("Conclusion: There is {} odds of acceptance for female applicants compared to male applicants, but this difference is not statistically significant given the model and data.".format(direction))
    else:
        desc_parts.append("Conclusion: Could not determine statistical significance from the available output.")

    description = " ".join(desc_parts)

    return {"object": numeric_result, "description": description}