def extract_final_answer(model_output):
    """
    Extract the effect of IsHomo from a fitted statsmodels GLMResultsWrapper (binomial logit).
    Returns a dict with:
      - "object": a dict of numeric results (coefficient, SE, z, p-value, 95% CI on log-odds,
                                  odds ratio, 95% CI on odds ratio, significance flag)
      - "description": a short interpretation in the context of whether modern humans
                       have higher AMTL than non-human primates after adjustment.
    """
    import numpy as np

    res = model_output

    # Basic safety checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = res.params
    # find the exact parameter name for IsHomo (allowing slight name mismatches)
    target_name = None
    if "IsHomo" in params.index:
        target_name = "IsHomo"
    else:
        # try to find a name that contains 'IsHomo' (case-sensitive)
        matches = [n for n in params.index if "IsHomo" in n]
        if len(matches) == 1:
            target_name = matches[0]
        elif len(matches) > 1:
            # if multiple matches, prefer exact match, else take first
            target_name = matches[0]

    if target_name is None:
        raise KeyError("Could not find a parameter named 'IsHomo' in model_output.params. Available params: {}".format(list(params.index)))

    coef = float(params[target_name])
    # standard error, z (or t) value, p-value, conf int
    se = float(res.bse[target_name]) if hasattr(res, "bse") else None
    # statsmodels GLM usually uses .tvalues (z-values) and .pvalues
    z_or_t = float(res.tvalues[target_name]) if hasattr(res, "tvalues") else None
    pval = float(res.pvalues[target_name]) if hasattr(res, "pvalues") else None
    conf = res.conf_int().loc[target_name] if hasattr(res, "conf_int") else None
    if conf is not None:
        ci_low, ci_high = float(conf[0]), float(conf[1])
    else:
        ci_low, ci_high = None, None

    # Convert log-odds to odds ratio
    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
    or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

    significant = (pval is not None) and (pval < 0.05)

    result_object = {
        "parameter_name": target_name,
        "coef_log_odds": coef,
        "std_err": se,
        "z_or_t": z_or_t,
        "p_value": pval,
        "ci_log_odds": [ci_low, ci_high],
        "odds_ratio": odds_ratio,
        "ci_odds_ratio": [or_ci_low, or_ci_high],
        "significant_at_0.05": significant
    }

    # Build a concise interpretation text
    if pval is None:
        interp = ("Extracted coefficient for '{}' = {:.4g}. Could not find p-value to assess significance."
                  .format(target_name, coef))
    else:
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        interp = (
            "Coefficient for '{}' = {:+.4f} (SE = {:.4f}, z/t = {:.3f}, p = {:.4g}). "
            "This corresponds to an odds ratio of {:.3f} (95% CI [{:.3f}, {:.3f}]). "
            "Interpretation: specimens coded as modern humans (IsHomo=1) have {} odds of antemortem "
            "tooth loss compared to non-human primates, after controlling for age, sex, and tooth class. "
            "The effect is {} at α = 0.05."
        ).format(
            target_name,
            coef,
            se if se is not None else float("nan"),
            z_or_t if z_or_t is not None else float("nan"),
            pval,
            odds_ratio,
            or_ci_low if or_ci_low is not None else float("nan"),
            or_ci_high if or_ci_high is not None else float("nan"),
            direction,
            "statistically significant" if significant else "not statistically significant"
        )

    return {"object": result_object, "description": interp}