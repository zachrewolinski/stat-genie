def extract_final_answer(model_output):
    """
    Extracts statistics on the effect of the 'female' indicator on mortgage denial
    from a statsmodels fit object contained in model_output.

    Returns a dictionary with keys:
      - "object": a dict with numeric results (coefficient, SE, p-value, CI, odds ratio, sample size,
                  and marginal effect if available)
      - "description": a short plain-English interpretation of the results
    """
    import math
    import numpy as np

    result = {
        "object": None,
        "description": None
    }

    if not isinstance(model_output, dict):
        result["description"] = "model_output must be a dict containing a fitted model under key 'fit'."
        return result

    fit = model_output.get("fit", None)
    n_obs = model_output.get("n_obs", None)

    if fit is None:
        result["description"] = "No fitted model found in model_output['fit']."
        return result

    # Try to extract coefficient, SE, p-value, and confidence interval for 'female'
    try:
        params = fit.params
    except Exception:
        # statsmodels wrapper should have .params; if not, bail out
        result["description"] = "Fitted object does not expose .params; cannot extract results."
        return result

    if "female" not in params.index:
        result["description"] = "The fitted model does not contain a parameter named 'female'."
        return result

    try:
        coef = float(params["female"])
    except Exception:
        coef = float(np.asarray(params["female"]))

    # Standard error and p-value (if available)
    try:
        se = float(fit.bse["female"])
    except Exception:
        se = None

    try:
        pvalue = float(fit.pvalues["female"])
    except Exception:
        pvalue = None

    # Confidence interval for the coefficient
    try:
        ci = fit.conf_int().loc["female"]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        ci_lower = None
        ci_upper = None

    # Odds ratio and CI on OR scale
    try:
        odds_ratio = math.exp(coef)
        or_ci_lower = math.exp(ci_lower) if ci_lower is not None else None
        or_ci_upper = math.exp(ci_upper) if ci_upper is not None else None
    except Exception:
        odds_ratio = None
        or_ci_lower = None
        or_ci_upper = None

    # Try to extract average marginal effect for 'female' if present
    marg_eff = None
    me_source = model_output.get("marginal_effects", None)
    if me_source is not None:
        try:
            # me_source may be a DataFrame-like object
            if "female" in me_source.index:
                # common column name for marginal effect is 'dy/dx'
                if "dy/dx" in me_source.columns:
                    marg_eff = float(me_source.loc["female", "dy/dx"])
                else:
                    # fallback: try first numeric column
                    numeric_cols = [c for c in me_source.columns if np.issubdtype(me_source[c].dtype, np.number)]
                    if numeric_cols:
                        marg_eff = float(me_source.loc["female", numeric_cols[0]])
        except Exception:
            marg_eff = None

    # Build the object to return
    object_dict = {
        "coef_female": coef,
        "se_female": se,
        "pvalue_female": pvalue,
        "ci_lower_female": ci_lower,
        "ci_upper_female": ci_upper,
        "odds_ratio_female": odds_ratio,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "marginal_effect_dydx_female": marg_eff,
        "n_obs": int(n_obs) if n_obs is not None else None
    }

    # Simple interpretation using conventional alpha = 0.05
    if pvalue is None:
        significance_text = "p-value not available; cannot assess statistical significance."
    else:
        significance_text = ("statistically significant at α=0.05"
                             if pvalue < 0.05
                             else "not statistically significant at α=0.05")

    # Direction interpretation
    if odds_ratio is None:
        direction_text = "Could not compute odds ratio."
    else:
        if odds_ratio > 1:
            direction_text = f"Females have higher odds of mortgage denial (OR = {odds_ratio:.3f})."
        elif odds_ratio < 1:
            direction_text = f"Females have lower odds of mortgage denial (OR = {odds_ratio:.3f})."
        else:
            direction_text = "No difference in odds of mortgage denial between females and males (OR ≈ 1)."

    # Assemble description
    desc_parts = []
    desc_parts.append(f"Estimated log-odds coefficient for 'female' = {coef:.4f}" if coef is not None else "Coefficient unavailable.")
    if se is not None:
        desc_parts.append(f"(SE = {se:.4f})")
    if pvalue is not None:
        desc_parts.append(f", p = {pvalue:.4g}.")
    else:
        desc_parts.append(".")
    if ci_lower is not None and ci_upper is not None:
        desc_parts.append(f"95% CI for coefficient: [{ci_lower:.4f}, {ci_upper:.4f}].")
    if odds_ratio is not None:
        desc_parts.append(f"Odds ratio = {odds_ratio:.3f}; 95% CI = [{or_ci_lower:.3f}, {or_ci_upper:.3f}].")
    desc_parts.append(direction_text)
    desc_parts.append(significance_text)
    if marg_eff is not None:
        desc_parts.append(f"Average marginal effect (dy/dx) for 'female' = {marg_eff:.4g} (on probability scale).")
    if n_obs is not None:
        desc_parts.append(f"Sample size used in model: {int(n_obs)} observations.")

    description = " ".join(desc_parts)

    result["object"] = object_dict
    result["description"] = description

    return result