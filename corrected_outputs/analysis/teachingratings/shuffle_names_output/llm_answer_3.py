def extract_final_answer(model_output):
    """
    Extracts the estimated effect of centered beauty (Beauty_c) on EvalScore from a fitted
    statsmodels RegressionResultsWrapper (with robust cov_type already applied).
    
    Returns a dictionary with:
      - "object": a dict containing numeric results (coef, se, t, p, 95% CI, nobs, significance)
      - "description": a short human-readable interpretation of the effect in context
    
    Requires that 'Beauty_c' is a parameter in model_output.params.
    """
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = res.params
    if "Beauty_c" not in params.index:
        raise ValueError("'Beauty_c' not found in model parameters. Check the fitted model.")

    # Extract stats for Beauty_c
    coef = float(params["Beauty_c"])
    # Use robust standard errors (bse should reflect the fitted cov_type)
    try:
        se = float(res.bse["Beauty_c"])
    except Exception:
        # fallback if bse is not available
        se = float(np.nan)
    try:
        tval = float(res.tvalues["Beauty_c"])
    except Exception:
        tval = float(np.nan)
    try:
        pval = float(res.pvalues["Beauty_c"])
    except Exception:
        pval = float(np.nan)

    # 95% confidence interval
    try:
        ci = res.conf_int(alpha=0.05).loc["Beauty_c"]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        ci_lower = float(np.nan)
        ci_upper = float(np.nan)

    # Sample size (observations used in the fit)
    try:
        nobs = int(res.nobs)
    except Exception:
        nobs = None

    significant = (pval < 0.05) if (not np.isnan(pval)) else None

    # Also extract quadratic term if present, to note potential nonlinearity
    quad_info = None
    if "Beauty_c2" in params.index:
        try:
            coef2 = float(params["Beauty_c2"])
            se2 = float(res.bse["Beauty_c2"])
            pval2 = float(res.pvalues["Beauty_c2"])
            ci2 = res.conf_int(alpha=0.05).loc["Beauty_c2"]
            ci2_lower = float(ci2[0]); ci2_upper = float(ci2[1])
            quad_info = {
                "coef": coef2,
                "se": se2,
                "p": pval2,
                "ci_lower": ci2_lower,
                "ci_upper": ci2_upper,
                "significant": (pval2 < 0.05)
            }
        except Exception:
            quad_info = None

    # Build numeric object to return
    obj = {
        "Beauty_c": {
            "coef": coef,
            "se": se,
            "t": tval,
            "p": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_at_0.05": significant
        },
        "Beauty_c2": quad_info,
        "nobs": nobs
    }

    # Construct a concise interpretation
    sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"
    desc_lines = [
        f"Estimated effect of centered beauty (Beauty_c) on course evaluation:",
        f"  Coefficient = {coef:.4f} (SE = {se:.4f}, t = {tval:.2f}, p = {pval:.4f}).",
        f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].",
        f"  Based on n = {nobs} observations, the effect is {sig_text}.",
    ]
    if quad_info is not None:
        qsig = "significant" if quad_info["significant"] else "not significant"
        desc_lines.append(
            f"Quadratic term Beauty_c2: coef = {quad_info['coef']:.4f}, p = {quad_info['p']:.4f} ({qsig})."
        )
        desc_lines.append(
            "If the quadratic term is significant it indicates a nonlinear (curved) relationship; otherwise the relationship is well summarized by the linear coefficient above."
        )

    # Plain-language interpretation
    if significant is True:
        desc_lines.append(
            "Interpretation: Holding controls constant, a one-unit increase in the centered beauty score is associated with"
            f" an average change of {coef:.4f} points in the course evaluation score (see CI above)."
        )
    elif significant is False:
        desc_lines.append(
            "Interpretation: There is no statistically reliable evidence that centered beauty is associated with course evaluation scores after adjusting for controls."
        )
    else:
        desc_lines.append("Interpretation: Significance could not be determined from available statistics.")

    description = " ".join(desc_lines)

    return {"object": obj, "description": description}