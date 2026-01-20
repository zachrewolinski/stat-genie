def extract_final_answer(model_output):
    """
    Extracts statistics for the 'DarkSkin' coefficient from the model output.

    Parameters
    ----------
    model_output : dict
        Expected to contain:
          - 'model_results': statsmodels-like results object (clustered robust results)
          - 'irr': pandas DataFrame with rows indexed by variable names and columns
                   including 'coef', 'IRR', 'IRR_lower_95', 'IRR_upper_95'

    Returns
    -------
    dict with keys:
      - "object": dict of extracted numeric results:
          * coef: estimated log rate ratio (coefficient)
          * se: clustered standard error for the coefficient (if available)
          * p_value: clustered p-value for the coefficient (if available)
          * IRR: incidence rate ratio = exp(coef)
          * IRR_lower_95, IRR_upper_95: 95% CI for the IRR
          * more_likely: boolean, True if IRR > 1
          * statistically_significant: boolean, True if p_value < 0.05 (or None if p-value not available)
      - "description": short textual interpretation in context
    """
    # Defensive checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'model_results' and 'irr'")

    results = model_output.get('model_results', None)
    irr_df = model_output.get('irr', None)

    # Prepare placeholders
    coef = se = p_value = irr = irr_lo = irr_hi = None

    # Try extracting from results object (clustered results)
    if results is not None:
        # params, bse, pvalues should be present for a statsmodels results object
        try:
            coef = float(results.params['DarkSkin'])
        except Exception:
            # fallback: try to get from irr df
            coef = None

        try:
            se = float(results.bse['DarkSkin'])
        except Exception:
            se = None

        try:
            p_value = float(results.pvalues['DarkSkin'])
        except Exception:
            p_value = None

    # Extract IRR and CI from irr dataframe if available
    if irr_df is not None:
        if 'DarkSkin' in irr_df.index:
            try:
                irr = float(irr_df.loc['DarkSkin', 'IRR'])
            except Exception:
                irr = None
            try:
                irr_lo = float(irr_df.loc['DarkSkin', 'IRR_lower_95'])
            except Exception:
                irr_lo = None
            try:
                irr_hi = float(irr_df.loc['DarkSkin', 'IRR_upper_95'])
            except Exception:
                irr_hi = None
            # If coef missing, attempt to take from irr_df['coef']
            if coef is None:
                try:
                    coef = float(irr_df.loc['DarkSkin', 'coef'])
                except Exception:
                    coef = None

    # If IRR missing but coef present, compute IRR
    try:
        if irr is None and coef is not None:
            irr = float(np.exp(coef))
    except Exception:
        pass

    # Determine simple inference
    more_likely = None
    if irr is not None:
        more_likely = irr > 1.0

    statistically_significant = None
    if p_value is not None:
        statistically_significant = (p_value < 0.05)

    # Build the object to return
    obj = {
        'coef_log_rate': coef,
        'std_error': se,
        'p_value': p_value,
        'IRR': irr,
        'IRR_lower_95': irr_lo,
        'IRR_upper_95': irr_hi,
        'more_likely': more_likely,
        'statistically_significant': statistically_significant
    }

    # Build a concise description
    desc_parts = []
    if coef is not None:
        desc_parts.append(f"DarkSkin coefficient (log rate): {coef:.4f}")
    if irr is not None:
        desc_parts.append(f"IRR = {irr:.3f}")
    if (irr_lo is not None) and (irr_hi is not None):
        desc_parts.append(f"95% CI for IRR = [{irr_lo:.3f}, {irr_hi:.3f}]")
    if p_value is not None:
        desc_parts.append(f"clustered p-value = {p_value:.3g}")
    if more_likely is not None:
        desc_parts.append("Direction: dark-skinned players are more likely to receive red cards" if more_likely else "Direction: dark-skinned players are less likely to receive red cards")
    if statistically_significant is not None:
        desc_parts.append("Statistically significant at alpha=0.05" if statistically_significant else "Not statistically significant at alpha=0.05")

    description = " ; ".join(desc_parts) if desc_parts else "Could not extract DarkSkin results from model_output."

    return {
        "object": obj,
        "description": description
    }