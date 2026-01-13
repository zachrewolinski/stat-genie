def extract_final_answer(model_output):
    """
    Extracts coefficient, uncertainty, and interpretation for 'Femininity_z'
    from the fitted deaths_model returned in model_output.

    Returns a dict with:
      - "object": a dict of extracted numeric results (coef, se, t, p, CI, percent-change)
      - "description": brief interpretation in context of the hypothesis
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict."
        }

    if 'deaths_model' not in model_output or model_output['deaths_model'] is None:
        return {
            "object": None,
            "description": "No 'deaths_model' found in model_output."
        }

    res = model_output['deaths_model']

    # Ensure the results object has necessary attributes
    needed_attrs = ['params', 'bse', 'tvalues', 'pvalues', 'conf_int']
    for a in needed_attrs:
        if not hasattr(res, a):
            return {
                "object": None,
                "description": f"The provided model result object is missing attribute '{a}'."
            }

    var = 'Femininity_z'
    try:
        coef = float(res.params[var])
        se = float(res.bse[var])
        tval = float(res.tvalues[var])
        pval = float(res.pvalues[var])
        ci = res.conf_int().loc[var].tolist()  # [lower, upper]
        ci_low = float(ci[0])
        ci_high = float(ci[1])
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract stats for variable '{var}': {e}"
        }

    # Transform coefficient on log(1 + deaths) to approximate percent change:
    # percent change ≈ exp(coef) - 1
    try:
        pct_change = (np.exp(coef) - 1.0) * 100.0
        ci_low_pct = (np.exp(ci_low) - 1.0) * 100.0
        ci_high_pct = (np.exp(ci_high) - 1.0) * 100.0
    except Exception:
        pct_change = None
        ci_low_pct = None
        ci_high_pct = None

    # Sample size if provided
    n_obs = model_output.get('n_obs_deaths_model', None)
    try:
        if n_obs is None and hasattr(res, 'nobs'):
            n_obs = int(res.nobs)
    except Exception:
        pass

    # Build object to return
    numeric_object = {
        "variable": var,
        "n_obs": int(n_obs) if n_obs is not None else None,
        "coef": coef,
        "se": se,
        "t": tval,
        "p": pval,
        "ci_95": [ci_low, ci_high],
        "approx_pct_change": pct_change,            # percent change in (1+deaths) per 1 SD increase in Femininity_z
        "ci_95_pct_change": [ci_low_pct, ci_high_pct]
    }

    # Simple significance summary
    if pval < 0.001:
        sig = "p < 0.001"
    elif pval < 0.01:
        sig = "p < 0.01"
    elif pval < 0.05:
        sig = "p < 0.05"
    else:
        sig = f"p = {pval:.3f} (not < 0.05)"

    # Interpretation: negative coef => more feminine name associated with fewer deaths
    direction = "negative" if coef < 0 else "positive" if coef > 0 else "null"
    interp_lines = [
        f"Estimate for '{var}': coef = {coef:.4f}, SE = {se:.4f}, t = {tval:.2f}, {sig}.",
        f"95% CI for coef = [{ci_low:.4f}, {ci_high:.4f}].",
    ]
    if pct_change is not None:
        interp_lines.append(
            f"On the original (log(1+deaths)) scale this corresponds to an approximate "
            f"{pct_change:.2f}% change in (1 + deaths) per 1 SD increase in name femininity "
            f"(95% CI: {ci_low_pct:.2f}% to {ci_high_pct:.2f}%)."
        )
    interp_lines.append(
        f"Direction: {direction} — a negative coefficient means more feminine hurricane "
        f"names are associated with fewer fatalities (supporting the hypothesis), "
        f"while a positive coefficient would indicate the opposite."
    )
    if n_obs is not None:
        interp_lines.append(f"Sample size used in model: {n_obs} hurricanes.")

    description = " ".join(interp_lines)

    return {
        "object": numeric_object,
        "description": description
    }