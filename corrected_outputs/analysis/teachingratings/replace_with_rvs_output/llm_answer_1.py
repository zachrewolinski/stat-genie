def extract_final_answer(model_output):
    """
    Extracts the estimated effect of instructor attractiveness (beauty_c) on teaching evaluations,
    including the interaction with instructor gender (gender_female). Returns a dictionary with:
      - "object": a dict of numeric results (coefficients, SEs, p-values, 95% CIs) for:
            * main effect (male instructors, gender_female=0)
            * interaction coefficient (beauty x female)
            * marginal effect for female instructors (main + interaction)
      - "description": short textual interpretation of the key numbers.

    The function is written to be robust to slightly different parameter name encodings
    (e.g., 'beauty_c:gender_female' vs 'beauty_c:gender_female').
    """
    import numpy as np
    try:
        from scipy import stats
        _have_scipy = True
    except Exception:
        _have_scipy = False

    res = model_output  # expected: statsmodels RegressionResultsWrapper or similar

    params = res.params
    pvalues = res.pvalues
    bse = res.bse
    cov = res.cov_params()

    # Identify parameter names robustly
    param_names = list(params.index.astype(str))

    # Find main beauty coefficient name (exact 'beauty_c' preferred)
    beauty_name = None
    if 'beauty_c' in param_names:
        beauty_name = 'beauty_c'
    else:
        # fallback: find a parameter that exactly equals or startswith 'beauty_c'
        for n in param_names:
            if n.split('[')[0] == 'beauty_c' or n.startswith('beauty_c'):
                beauty_name = n
                break
    if beauty_name is None:
        # as last resort, pick any parameter that contains 'beauty' and 'c'
        for n in param_names:
            if 'beaut' in n and 'c' in n:
                beauty_name = n
                break

    # Find interaction name: contains both beauty and gender_female
    interaction_name = None
    for n in param_names:
        if 'beauty' in n and 'gender_female' in n:
            interaction_name = n
            break
    # Also consider the reverse order (gender_female:beauty_c)
    if interaction_name is None:
        for n in param_names:
            if 'gender_female' in n and 'beauty' in n:
                interaction_name = n
                break

    if beauty_name is None:
        raise KeyError("Could not find a parameter name for the main effect 'beauty_c' in model parameters: %s" % str(param_names))

    # Extract main effect (this is the effect for gender_female = 0, i.e., male instructors)
    beta_beauty = float(params[beauty_name])
    # bse may be a Series with key beauty_name or we can fallback to sqrt of covariance diagonal
    try:
        se_beauty = float(bse.get(beauty_name, np.sqrt(cov.loc[beauty_name, beauty_name])))
    except Exception:
        se_beauty = float(np.sqrt(float(cov.loc[beauty_name, beauty_name])))
    p_beauty = float(pvalues.get(beauty_name, np.nan))
    # CI for main effect (using t with df_resid if available)
    df = getattr(res, 'df_resid', None)
    if _have_scipy and df is not None:
        tcrit = stats.t.ppf(0.975, df)
    elif _have_scipy:
        tcrit = stats.norm.ppf(0.975)
    else:
        # approximate normal critical value if scipy not available
        tcrit = 1.959963984540054

    ci_beauty = [beta_beauty - tcrit * se_beauty, beta_beauty + tcrit * se_beauty]

    # Interaction (may be None)
    if interaction_name is not None and interaction_name in params.index:
        beta_int = float(params[interaction_name])
        try:
            se_int = float(bse.get(interaction_name, np.sqrt(cov.loc[interaction_name, interaction_name])))
        except Exception:
            se_int = float(np.sqrt(float(cov.loc[interaction_name, interaction_name])))
        p_int = float(pvalues.get(interaction_name, np.nan))
        ci_int = [beta_int - tcrit * se_int, beta_int + tcrit * se_int]
    else:
        beta_int = 0.0
        se_int = 0.0
        p_int = np.nan
        ci_int = [np.nan, np.nan]

    # Marginal effect for female instructors = beta_beauty + beta_int
    beta_female = beta_beauty + beta_int
    # Var(beta_female) = Var(beta_beauty) + Var(beta_int) + 2*Cov(beta_beauty, beta_int)
    if interaction_name is not None and interaction_name in cov.index and beauty_name in cov.index:
        var_beauty = cov.loc[beauty_name, beauty_name]
        var_int = cov.loc[interaction_name, interaction_name]
        cov_bi = cov.loc[beauty_name, interaction_name]
        var_female = var_beauty + var_int + 2.0 * cov_bi
        se_female = float(np.sqrt(var_female)) if var_female >= 0 else float(np.nan)
    else:
        # If no interaction term or covariance unavailable, marginal effect equals main effect
        se_female = se_beauty

    # p-value for female marginal effect
    if (not np.isnan(se_female)) and (se_female > 0):
        t_female = beta_female / se_female
        if _have_scipy and df is not None:
            p_female = float(2.0 * (1.0 - stats.t.cdf(abs(t_female), df)))
        elif _have_scipy:
            p_female = float(2.0 * (1.0 - stats.norm.cdf(abs(t_female))))
        else:
            # approximate via normal using math.erfc
            from math import erfc, sqrt
            p_female = 2.0 * 0.5 * erfc(abs(t_female) / sqrt(2.0))
    else:
        p_female = np.nan

    ci_female = [beta_female - tcrit * se_female, beta_female + tcrit * se_female] if (not np.isnan(se_female)) else [np.nan, np.nan]

    # Build return object
    # Convert params and bse to plain dicts (they may be pandas Series)
    try:
        raw_params = params.to_dict()
    except Exception:
        raw_params = dict(params)
    try:
        raw_bse = bse.to_dict()
    except Exception:
        raw_bse = dict(bse)

    result_object = {
        "beauty_name": beauty_name,
        "interaction_name": interaction_name,
        "male": {
            "coef": beta_beauty,
            "se": se_beauty,
            "p_value": p_beauty,
            "95%_CI": ci_beauty,
            "interpretation": "Effect of a 1-unit increase in centered attractiveness on eval for male instructors (gender_female=0)."
        },
        "female": {
            "coef": beta_female,
            "se": se_female,
            "p_value": p_female,
            "95%_CI": ci_female,
            "interpretation": "Marginal effect of a 1-unit increase in centered attractiveness on eval for female instructors (gender_female=1). Computed as main beauty effect + beauty x female interaction."
        },
        "interaction": {
            "coef": beta_int,
            "se": se_int,
            "p_value": p_int,
            "95%_CI": ci_int,
            "interpretation": "How much the effect of beauty differs for female instructors relative to male instructors."
        },
        # For transparency, include the raw parameter estimates for reference:
        "raw_params": raw_params,
        "raw_bse": raw_bse,
    }

    # Short textual description summarizing whether beauty matters
    # We consider it "statistically significant" if p < 0.05 for the marginal effect.
    sig_male = (not np.isnan(p_beauty)) and (p_beauty < 0.05)
    sig_female = (not np.isnan(p_female)) and (p_female < 0.05)

    # Safely format numeric values for the description
    def fmt(x):
        try:
            if x is None:
                return "None"
            if np.isnan(x):
                return "nan"
            return f"{x:.3f}"
        except Exception:
            return str(x)

    if interaction_name is not None:
        if sig_male and sig_female:
            desc = (
                "The estimated effect of attractiveness (beauty) on teaching evaluations is "
                f"{fmt(beta_beauty)} (SE={fmt(se_beauty)}, p={fmt(p_beauty)}) for male instructors, "
                f"and {fmt(beta_female)} (SE={fmt(se_female)}, p={fmt(p_female)}) for female instructors. "
                f"The interaction (difference) is {fmt(beta_int)} (SE={fmt(se_int)}, p={fmt(p_int)})."
            )
        else:
            desc = (
                "The estimated effect of attractiveness on teaching evaluations is "
                f"{fmt(beta_beauty)} (SE={fmt(se_beauty)}, p={fmt(p_beauty)}) for male instructors; "
                "for female instructors the marginal effect is "
                f"{fmt(beta_female)} (SE={fmt(se_female)}, p={fmt(p_female)}). "
                f"The interaction coefficient is {fmt(beta_int)} (SE={fmt(se_int)}, p={fmt(p_int)})."
            )
    else:
        # No interaction term present
        desc = (
            "The model contains no beauty x gender interaction. The effect of attractiveness on teaching "
            "evaluations (pooled / for the reference gender) is "
            f"{fmt(beta_beauty)} (SE={fmt(se_beauty)}, p={fmt(p_beauty)})."
        )

    return {"object": result_object, "description": desc}