def extract_final_answer(model_output):
    """
    Extract statistics for the 'IsHuman' effect from a fitted statsmodels GLMResultsWrapper.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coefficient, SE, z, p, 95% CI,
                  odds ratio and its 95% CI, boolean 'significant' at alpha=0.05).
      - "description": a short interpretation about whether modern humans show higher
                       AMTL than non-human primates after accounting for controls.

    The function is defensive: it searches for a parameter name containing 'IsHuman'
    in the model parameter index and will raise a clear error if not found.
    """
    import numpy as np

    # Basic validation
    if model_output is None:
        raise ValueError("model_output is None")

    # Ensure required attributes exist
    for attr in ("params", "bse", "pvalues", "conf_int"):
        if not hasattr(model_output, attr):
            raise AttributeError(f"model_output missing required attribute '{attr}'")

    params = model_output.params
    bse = model_output.bse
    pvalues = model_output.pvalues
    try:
        ci = model_output.conf_int()  # DataFrame or array: rows correspond to params
    except Exception:
        # statsmodels sometimes exposes conf_int as a method with args
        ci = model_output.conf_int()

    # Find the parameter name corresponding to IsHuman (be permissive)
    param_names = list(params.index)
    match_name = None
    for name in param_names:
        if 'IsHuman' == name or name.endswith('.IsHuman') or 'IsHuman' in name:
            match_name = name
            break
    if match_name is None:
        raise KeyError("Could not find a parameter corresponding to 'IsHuman' in model parameters. "
                       f"Available parameters: {param_names}")

    coef = float(params[match_name])
    se = float(bse[match_name])
    # compute z (Wald z-statistic) and p if not trustworthy from pvalues
    z = coef / se if se != 0 else np.nan
    p = float(pvalues[match_name])
    ci_row = ci.loc[match_name] if hasattr(ci, "loc") else ci[param_names.index(match_name)]
    ci_lower = float(ci_row[0])
    ci_upper = float(ci_row[1])

    # For a binomial GLM with logit link, exponentiating coefficient gives odds ratio
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    alpha = 0.05
    significant = (p < alpha)

    # Short interpretation: positive coef => higher log-odds (hence higher probability) of AMTL for humans
    if np.isnan(coef):
        interpretation = "Coefficient for IsHuman is NaN; cannot interpret."
    else:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        if significant:
            interpretation = (
                f"The model estimates that modern humans (IsHuman=1) have {direction} AMTL than "
                "non-human primates, controlling for age, sex, and tooth class. "
                f"Coef={coef:.4f} (SE={se:.4f}), z={z:.2f}, p={p:.3g}. "
                f"Odds ratio={odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
            )
        else:
            interpretation = (
                f"The model does not provide statistically significant evidence (alpha={alpha}) that "
                f"modern humans differ in AMTL from non-human primates after controlling for covariates. "
                f"Estimated coef={coef:.4f} (SE={se:.4f}), z={z:.2f}, p={p:.3g}. "
                f"Odds ratio={odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
            )

    result_object = {
        "parameter_name": match_name,
        "coef": coef,
        "std_error": se,
        "z_value": z,
        "p_value": p,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "odds_ratio": odds_ratio,
        "or_ci_95_lower": or_ci_lower,
        "or_ci_95_upper": or_ci_upper,
        "significant_at_0.05": bool(significant),
        "alpha": alpha,
    }

    return {"object": result_object, "description": interpretation}