def extract_final_answer(model_output):
    """
    Extracts the coefficient and inference for the 'HasChildren' variable
    from the model_output produced by the provided Tobit-fitting function.

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, se, z, pval, 95% CI, sigma)
      - "description": a short plain-language interpretation answering whether
                       having children decreases engagement in extramarital affairs.
    """
    import numpy as np
    import pandas as pd

    # Expect a pandas DataFrame under 'params_table'
    params_table = model_output.get('params_table')
    if params_table is None or not isinstance(params_table, pd.DataFrame):
        raise ValueError("model_output must contain a pandas DataFrame under key 'params_table'.")

    if 'HasChildren' not in params_table.index:
        raise ValueError("The parameters table does not contain a row for 'HasChildren'.")

    row = params_table.loc['HasChildren']
    # Extract values if present; allow for missing columns gracefully
    coef = float(row.get('param', np.nan))
    se = float(row.get('se', np.nan))
    z = float(row.get('z', np.nan)) if 'z' in row.index else (coef / se if se and not np.isnan(se) else np.nan)
    pval = float(row.get('pval', np.nan))

    # 95% Wald/confidence interval using normal approx
    if not np.isnan(se):
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
    else:
        ci_lower = ci_upper = np.nan

    # Try to get sigma (on original scale) if provided in the table
    sigma = None
    if 'sigma' in params_table.index:
        try:
            sigma = float(params_table.loc['sigma', 'param'])
        except Exception:
            sigma = None
    elif 'log_sigma' in params_table.index:
        try:
            log_sigma = float(params_table.loc['log_sigma', 'param'])
            sigma = float(np.exp(log_sigma))
        except Exception:
            sigma = None

    # Build numeric object to return
    numeric_result = {
        'coef_HasChildren': coef,
        'se_HasChildren': se,
        'z_HasChildren': z,
        'pval_HasChildren': pval,
        '95ci_HasChildren': (ci_lower, ci_upper),
        'sigma': sigma
    }

    # Interpretation targeted to the user's question
    # Note: Tobit coefficient is on the latent (uncensored) outcome.
    if not np.isnan(pval) and pval < 0.05:
        significance_statement = "statistically significant"
    else:
        significance_statement = "not statistically significant"

    description = (
        f"The estimated Tobit coefficient for HasChildren is {coef:.3f} "
        f"(SE={se:.3f}, z={z:.3f}, p={pval:.3f}; 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
        f"This coefficient is {significance_statement}. "
        "Because this is a Tobit model, the coefficient refers to the effect on the latent "
        "propensity/underlying affair-frequency measure (not directly the observed mean frequency), "
        f"and the estimated residual sd is approximately {sigma if sigma is not None else 'N/A'}. "
    )

    # Direct answer to the user's yes/no question:
    # The point estimate is positive (suggesting a slight increase in affair-frequency for those with children),
    # but it is not statistically significant, so there is no evidence that having children decreases engagement in extramarital affairs.
    if not np.isnan(pval) and pval < 0.05:
        # significant: check sign
        if coef < 0:
            direct_answer = "Yes — having children is associated with a statistically significant decrease in affair frequency."
        else:
            direct_answer = "No — having children is associated with a statistically significant increase in affair frequency."
    else:
        # not significant
        if coef < 0:
            direct_answer = ("No strong evidence that having children decreases affair engagement. "
                             "Point estimate suggests a decrease but it is not statistically significant.")
        elif coef > 0:
            direct_answer = ("No — there is no evidence that having children decreases affair engagement. "
                             "The point estimate actually suggests a small increase, but it is not statistically significant.")
        else:
            direct_answer = "No evidence of any effect of having children on affair engagement."

    description += " " + direct_answer

    return {
        "object": numeric_result,
        "description": description
    }