def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of the 'Female' indicator on mortgage approval
    from the model_output produced by the modeling function.

    Returns a dictionary with:
      - "object": dict of numeric results (coef, p_value, odds_ratio, CI, significance, n_obs)
      - "description": human-readable interpretation of the Female effect
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict (the output from the model function).")

    res = model_output.get('result', None)
    or_table = model_output.get('odds_ratios', None)

    if res is None:
        raise ValueError("No 'result' entry found in model_output.")

    # Try to extract coefficient, p-value and CI from the statsmodels result
    coef = None
    p_value = None
    ci_lower = None
    ci_upper = None
    odds_ratio = None

    try:
        # Prefer extracting from the fitted result (log-odds scale)
        coef = float(res.params.loc['Female'])
        p_value = float(res.pvalues.loc['Female']) if 'Female' in res.pvalues.index else None
        ci_log = res.conf_int().loc['Female']  # (lower_log, upper_log)
        ci_lower = float(np.exp(ci_log[0]))
        ci_upper = float(np.exp(ci_log[1]))
        odds_ratio = float(np.exp(coef))
    except Exception:
        # Fallback: use the provided odds_ratios table if available
        if or_table is not None and 'Female' in or_table.index:
            row = or_table.loc['Female']
            odds_ratio = float(row['OR'])
            ci_lower = float(row['CI_lower'])
            ci_upper = float(row['CI_upper'])
            try:
                coef = float(np.log(odds_ratio))
            except Exception:
                coef = None
            p_value = None  # p-value not available from the OR table
        else:
            raise ValueError("Couldn't extract statistics for 'Female' from the model output.")

    # Sample size
    n_obs = model_output.get('n_obs', None)
    # try alternative sources
    if n_obs is None:
        try:
            n_obs = int(getattr(res, 'nobs', None))
        except Exception:
            n_obs = None

    # Determine significance if p-value available
    significant = None
    alpha = 0.05
    if p_value is not None:
        significant = (p_value < alpha)

    # Build human-readable description
    desc_parts = []
    if odds_ratio is not None and ci_lower is not None and ci_upper is not None:
        desc_parts.append(
            f"Controlling for included covariates, being female is associated with an odds ratio (OR) = {odds_ratio:.3f} "
            f"(95% CI: {ci_lower:.3f}–{ci_upper:.3f})."
        )
    elif odds_ratio is not None:
        desc_parts.append(f"Estimated OR for females (vs males) = {odds_ratio:.3f}.")

    if coef is not None:
        desc_parts.append(f"Log-odds coefficient = {coef:.3f}.")
    if p_value is not None:
        desc_parts.append(f"p-value = {p_value:.3g}.")
        if significant:
            desc_parts.append(f"This effect is statistically significant at alpha = {alpha}.")
        else:
            desc_parts.append(f"This effect is not statistically significant at alpha = {alpha}.")
    else:
        desc_parts.append("p-value not available from the provided output.")

    if n_obs is not None:
        desc_parts.append(f"Sample size (observations used) = {int(n_obs)}.")

    # Numeric object to return
    object_out = {
        'variable': 'Female',
        'coef_log_odds': None if coef is None else float(coef),
        'p_value': None if p_value is None else float(p_value),
        'odds_ratio': None if odds_ratio is None else float(odds_ratio),
        'ci_lower': None if ci_lower is None else float(ci_lower),
        'ci_upper': None if ci_upper is None else float(ci_upper),
        'significant_at_0.05': significant,
        'n_obs': None if n_obs is None else int(n_obs)
    }

    description = " ".join(desc_parts)

    return {
        "object": object_out,
        "description": description
    }