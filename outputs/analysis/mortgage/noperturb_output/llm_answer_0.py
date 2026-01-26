def extract_final_answer(model_output):
    """
    Extracts statistics for the 'Female' coefficient from a fitted statsmodels model
    (GLMResultsWrapper, LogitResults, or similar).

    Returns a dictionary with:
      - "object": dict containing numeric results:
            {
              'coef': float,
              'se': float or None,
              'p_value': float,
              'ci_lower': float,
              'ci_upper': float,
              'odds_ratio': float,
              'odds_ratio_ci_lower': float,
              'odds_ratio_ci_upper': float,
              'significant_0.05': bool
            }
      - "description": human-readable interpretation of the coefficient in context.
    """
    import numpy as np
    import pandas as pd

    # Extract parameter-related arrays/Series robustly
    try:
        params = getattr(model_output, "params")
        bse = getattr(model_output, "bse", None)
        pvalues = getattr(model_output, "pvalues", None)
        conf = None
        # conf_int may be a method or attribute
        if hasattr(model_output, "conf_int"):
            conf = model_output.conf_int()
    except Exception as e:
        raise ValueError(f"Provided model_output does not appear to be a statsmodels results object: {e}")

    # Ensure params is a pandas Series (convert if ndarray)
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params, index=getattr(model_output, "model").exog_names)
        except Exception:
            params = pd.Series(params)

    # Convert bse and pvalues to Series with matching index if they are present
    if bse is None:
        bse_series = None
    else:
        if not isinstance(bse, pd.Series):
            try:
                bse_series = pd.Series(bse, index=params.index)
            except Exception:
                bse_series = pd.Series(bse)
        else:
            bse_series = bse

    if pvalues is None:
        raise ValueError("Model output does not contain p-values (model_output.pvalues missing).")
    else:
        if not isinstance(pvalues, pd.Series):
            try:
                pvalues = pd.Series(pvalues, index=params.index)
            except Exception:
                pvalues = pd.Series(pvalues)

    # Normalize conf to a DataFrame with same index as params
    if conf is None:
        raise ValueError("Model output does not provide confidence intervals (conf_int missing).")
    if not isinstance(conf, pd.DataFrame):
        try:
            conf = pd.DataFrame(conf, index=params.index)
        except Exception:
            # As a last resort, try to map rows to param order
            conf = pd.DataFrame(conf)

    # Check 'Female' present
    if 'Female' not in params.index:
        raise ValueError("The model does not have a 'Female' coefficient in its parameters.")

    # Extract statistics for 'Female'
    coef = float(params.loc['Female'])
    se = float(bse_series.loc['Female']) if (bse_series is not None and 'Female' in bse_series.index) else None
    p_value = float(pvalues.loc['Female'])

    # conf DataFrame may have two columns; take them by position
    try:
        ci_row = conf.loc['Female']
        ci_lower = float(ci_row.iloc[0])
        ci_upper = float(ci_row.iloc[1])
    except Exception:
        # fallback: if conf has numeric index positions
        try:
            female_idx = list(params.index).index('Female')
            ci_lower = float(conf.iloc[female_idx, 0])
            ci_upper = float(conf.iloc[female_idx, 1])
        except Exception as e:
            raise ValueError(f"Could not extract confidence interval for 'Female': {e}")

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    significant = bool(p_value < 0.05)

    # Short interpretation
    if significant:
        if odds_ratio > 1:
            effect_desc = "Being female is associated with higher odds of mortgage acceptance"
        else:
            effect_desc = "Being female is associated with lower odds of mortgage acceptance"
    else:
        effect_desc = "No statistically significant association between being female and mortgage acceptance at alpha=0.05"

    description = (
        f"Female coefficient = {coef:.4f}"
        + (f" (SE = {se:.4f})" if se is not None else "")
        + f", p = {p_value:.3g}. 95% CI for coefficient = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        + f"Odds ratio = {odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]). "
        + f"Significant at alpha=0.05: {significant}. {effect_desc}. "
        + "Estimates control for Black, Married, SelfEmployed, BadHistory, DeniedPMI, MortgageCredit_z, "
        + "ConsumerCredit_z, PI_ratio_z, LoanToValue_z, and HousingExpenseRatio_z."
    )

    result_object = {
        'coef': coef,
        'se': se,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'odds_ratio': odds_ratio,
        'odds_ratio_ci_lower': or_ci_lower,
        'odds_ratio_ci_upper': or_ci_upper,
        'significant_0.05': significant
    }

    return {"object": result_object, "description": description}