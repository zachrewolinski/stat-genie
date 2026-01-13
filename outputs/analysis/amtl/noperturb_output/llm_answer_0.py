def extract_final_answer(model_output):
    """
    Extracts the effect of 'is_human' from the model output and interprets whether
    modern humans have higher AMTL after accounting for covariates.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, z, p, odds_ratio, or_ci_lower, or_ci_upper,
                  significant (bool), decision_text)
      - "description": brief plain-language interpretation of the result in context
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    # Prefer clustered results (robust to within-specimen correlation); fallback to glm_results
    clustered = model_output.get('clustered_results') or model_output.get('glm_results')
    results = model_output.get('glm_results')

    # Obtain parameter vector
    if not hasattr(clustered, 'params'):
        raise ValueError("Provided model_output does not contain params in clustered_results or glm_results.")

    params = clustered.params

    if 'is_human' not in params.index:
        raise KeyError("'is_human' not found in model parameters.")

    # Try to get standard errors, confidence intervals from clustered object
    # ClusteredResults created in the model provides: cov_params(), bse, conf_int()
    try:
        # bse may be an array or Series; normalize to Series
        bse = clustered.bse
        if not isinstance(bse, pd.Series):
            bse = pd.Series(bse, index=params.index)
    except Exception:
        # Fallback: attempt to compute SE from covariance matrix if available
        try:
            cov = clustered.cov_params()
            se_vals = np.sqrt(np.diag(cov))
            bse = pd.Series(se_vals, index=params.index)
        except Exception:
            bse = None

    try:
        conf = clustered.conf_int()
    except Exception:
        conf = None

    coef = float(params['is_human'])
    se = float(bse['is_human']) if bse is not None else None

    # z and p-value (use normal approximation for clustered sandwich SE)
    if se is not None and se > 0:
        z = coef / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    else:
        z = None
        p_value = None

    # Odds ratio and CI (on exponentiated scale)
    odds_ratio = float(np.exp(coef))
    if conf is not None:
        # conf expected as DataFrame with columns [0,1]
        try:
            ci_lower = float(np.exp(conf.loc['is_human', 0]))
            ci_upper = float(np.exp(conf.loc['is_human', 1]))
        except Exception:
            # fallback if indexing differs
            row = conf.loc['is_human']
            ci_lower = float(np.exp(row.iloc[0]))
            ci_upper = float(np.exp(row.iloc[1]))
    else:
        # approximate 95% CI using coef +/- 1.96*se on log-odds scale
        if se is not None:
            ci_lower = float(np.exp(coef - 1.96 * se))
            ci_upper = float(np.exp(coef + 1.96 * se))
        else:
            ci_lower = None
            ci_upper = None

    # Decision: higher AMTL if OR > 1 and p < 0.05
    significant = (p_value is not None) and (p_value < 0.05)
    higher = (odds_ratio > 1) and significant

    # Short interpretation text
    if significant:
        decision_text = (
            "Yes — after adjusting for age, sex probability, and tooth class, modern humans "
            "have significantly higher odds of antemortem tooth loss compared to the included "
            "non-human primates."
        )
    else:
        decision_text = (
            "No — there is not strong evidence that modern humans differ from the non-human "
            "primates in AMTL after adjusting for the covariates."
        )

    # Include dispersion if available as contextual caution (overdispersion may affect inference)
    dispersion = model_output.get('dispersion', None)

    output_obj = {
        'coef_logit_is_human': coef,
        'se_logit_is_human': se,
        'z_value': z,
        'p_value': p_value,
        'odds_ratio_is_human': odds_ratio,
        'or_ci_lower_95': ci_lower,
        'or_ci_upper_95': ci_upper,
        'significant_at_0.05': significant,
        'decision_higher_amtl': higher,
        'dispersion': float(dispersion) if dispersion is not None else None,
        'decision_text': decision_text
    }

    # Human-readable description
    description_lines = []
    description_lines.append(
        f"Estimated odds ratio for is_human = {odds_ratio:.3f} "
        f"(95% CI: {ci_lower:.3f} – {ci_upper:.3f})"
        if (ci_lower is not None and ci_upper is not None) else
        f"Estimated odds ratio for is_human = {odds_ratio:.3f} (CI unavailable)"
    )
    if p_value is not None:
        description_lines.append(f"Two-sided p-value (Wald z) = {p_value:.3g}.")
    if dispersion is not None:
        description_lines.append(f"Model dispersion = {dispersion:.3f} (values >>1 suggest overdispersion).")
    description_lines.append(decision_text)
    description = " ".join(description_lines)

    return {'object': output_obj, 'description': description}