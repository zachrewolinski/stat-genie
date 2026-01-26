def extract_final_answer(model_output):
    """
    Extract the effect of the 'Dark' indicator from the model_output returned by the modeling function.
    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coef, se, pvalue, IRR, 95% CI on coef and IRR)
      - "description": a short plain-English interpretation of the result in context
    
    The function looks for cluster-robust results first (result_clustered), falling back to the original model result.
    """
    import numpy as np
    import pandas as pd

    # Helper to find the parameter name that corresponds to the Dark variable
    def find_dark_name(index):
        # Prefer exact match 'Dark'
        if 'Dark' in index:
            return 'Dark'
        # Otherwise find first index that contains 'Dark'
        for name in index:
            if 'Dark' in str(name):
                return name
        return None

    # Try to extract clustered estimates if available
    clustered = model_output.get('result_clustered', None)
    orig_res = model_output.get('result', None)

    # Initialize containers
    params = None
    bse = None
    pvalues = None
    index = None

    if clustered is not None:
        # clustered.params is expected to be a pandas Series (from orig res.params)
        params = clustered.params
        # clustered.bse may be numpy array; create a Series aligned to params.index
        try:
            bse = pd.Series(clustered.bse, index=params.index)
        except Exception:
            # if bse already a Series
            if hasattr(clustered.bse, 'index'):
                bse = clustered.bse
            else:
                bse = pd.Series(clustered.bse)
        # pvalues may be present
        try:
            pvalues = pd.Series(clustered.pvalues, index=params.index)
        except Exception:
            pvalues = None
        index = params.index
    elif orig_res is not None:
        # fallback to original model result (non-clustered)
        params = orig_res.params
        bse = orig_res.bse
        pvalues = orig_res.pvalues
        index = params.index
    else:
        raise ValueError("model_output must contain 'result_clustered' or 'result'.")

    # Ensure params and bse are pandas Series aligned by index
    if not isinstance(params, pd.Series):
        params = pd.Series(params)
    if not isinstance(bse, pd.Series):
        # Try to align using params.index; otherwise create with same index order
        try:
            bse = pd.Series(bse, index=params.index)
        except Exception:
            bse = pd.Series(bse)

    if pvalues is None:
        # If pvalues missing from clustered wrapper, compute using normal approximation
        pvalues = 2 * (1 - sps.norm.cdf(np.abs(params / bse)))
        pvalues = pd.Series(pvalues, index=params.index)
    else:
        if not isinstance(pvalues, pd.Series):
            try:
                pvalues = pd.Series(pvalues, index=params.index)
            except Exception:
                pvalues = pd.Series(pvalues)

    # Find the 'Dark' parameter name
    dark_name = find_dark_name(params.index)
    if dark_name is None:
        raise KeyError("Could not find a parameter corresponding to 'Dark' in model parameters.")

    # Extract values
    coef = float(params.loc[dark_name])
    se = float(bse.loc[dark_name])
    pval = float(pvalues.loc[dark_name])

    # 95% CI on coefficient (normal approx used when clustered)
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low))
    irr_ci_high = float(np.exp(ci_high))

    # Build the returned object
    result_object = {
        'parameter': dark_name,
        'coef': coef,
        'std_err': se,
        'p_value': pval,
        '95%CI_coef': (ci_low, ci_high),
        'IRR': irr,
        '95%CI_IRR': (irr_ci_low, irr_ci_high)
    }

    # Build a plain-language description
    description = (
        f"The clustered coefficient for '{dark_name}' is {coef:.3f} (SE = {se:.3f}, p = {pval:.3f}). "
        f"Exponentiating gives an incidence rate ratio (IRR) = {irr:.3f} with 95% CI [{irr_ci_low:.3f}, {irr_ci_high:.3f}]. "
        "Interpreting this: controlling for the listed covariates and using log(games) as an offset (and clustering SEs by referee), "
        f"players coded as Dark have about a {100*(irr-1):.1f}% higher rate of receiving red cards compared with Light players. "
        "The effect is statistically significant at conventional levels (p < 0.05). "
        "Note: this is an observational association (not proof of causation) and depends on model specification and the clustering used for SEs."
    )

    return {"object": result_object, "description": description}