def extract_final_answer(model_output):
    """
    Extract statistics for the key predictor (SkinDark) from the model output
    and return a concise interpretation.

    Returns a dict with:
      - "object": dict with numeric results {coef, IRR, IRR_CI, pvalue, significant, conclusion}
      - "description": short human-readable explanation of what the numbers mean
    """
    import numpy as np
    import pandas as pd

    # Initialize placeholders
    coef = None
    se = None
    pvalue = None
    irr = None
    irr_ci_lower = None
    irr_ci_upper = None

    # Accept either the dict returned by the modeling function or the raw results object
    res = None
    irr_table = None

    # If model_output is a dict with expected keys, pull them
    if isinstance(model_output, dict):
        if 'results_object' in model_output:
            res = model_output['results_object']
        if 'irr_table' in model_output:
            irr_table = model_output['irr_table']
    else:
        # If a single results object was passed directly
        res = model_output

    # Try to extract from irr_table first (convenient already-transformed values)
    try:
        if irr_table is not None:
            # Ensure it's a DataFrame or similar
            if 'SkinDark' in irr_table.index:
                row = irr_table.loc['SkinDark']
                irr = float(row.get('IRR', np.nan))
                irr_ci_lower = float(row.get('IRR_CI_lower', np.nan))
                irr_ci_upper = float(row.get('IRR_CI_upper', np.nan))
                pvalue = float(row.get('pvalue', np.nan))
    except Exception:
        # ignore and fall back to results object extraction
        irr_table = None

    # If irr_table didn't yield values, try extracting from the results_object
    if irr is None or (isinstance(irr, float) and np.isnan(irr)):
        if res is not None:
            # Try to access params, pvalues, conf_int
            try:
                params = res.params
                pvalues = res.pvalues
                conf = res.conf_int()
                # Coefficient (log scale)
                if 'SkinDark' in params.index:
                    coef = float(params['SkinDark'])
                    pvalue = float(pvalues['SkinDark'])
                    # IRR and CI by exponentiating
                    irr = float(np.exp(coef))
                    try:
                        ci_lower = float(conf.loc['SkinDark'].iat[0])
                        ci_upper = float(conf.loc['SkinDark'].iat[1])
                        irr_ci_lower = float(np.exp(ci_lower))
                        irr_ci_upper = float(np.exp(ci_upper))
                    except Exception:
                        # if conf is not indexed by name or has different layout
                        try:
                            # assume same order as params
                            idx = list(params.index).index('SkinDark')
                            ci_lower = float(conf.iloc[idx, 0])
                            ci_upper = float(conf.iloc[idx, 1])
                            irr_ci_lower = float(np.exp(ci_lower))
                            irr_ci_upper = float(np.exp(ci_upper))
                        except Exception:
                            irr_ci_lower = np.nan
                            irr_ci_upper = np.nan
                else:
                    # Couldn't find SkinDark in params
                    pass
            except Exception:
                # any extraction error -> leave values as None/NaN
                pass

    # Build conclusion
    significant = False
    conclusion = "Could not determine effect for SkinDark (variable not found)."
    if pvalue is not None and not (np.isnan(pvalue)):
        significant = (pvalue < 0.05)
        if np.isnan(irr):
            conclusion = "Coefficient found but could not compute IRR."
        else:
            if significant:
                if irr > 1:
                    conclusion = ("Statistically significant evidence (p = {:.3g}) that players coded as dark-skinned "
                                  "receive red cards at a higher rate than light-skinned players (IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}])."
                                  ).format(pvalue, irr, irr_ci_lower, irr_ci_upper)
                else:
                    conclusion = ("Statistically significant evidence (p = {:.3g}) that players coded as dark-skinned "
                                  "receive red cards at a lower rate than light-skinned players (IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}])."
                                  ).format(pvalue, irr, irr_ci_lower, irr_ci_upper)
            else:
                conclusion = ("No statistically significant difference between dark- and light-skinned players in red-card "
                              "rates (p = {:.3g}). Estimated IRR = {:.3f} with 95% CI [{:.3f}, {:.3f}], which is compatible with "
                              "no meaningful difference.").format(pvalue, irr, irr_ci_lower, irr_ci_upper)

    # Prepare the object to return
    result_object = {
        'coef_log': None if coef is None else float(coef),
        'IRR': None if irr is None else float(irr),
        'IRR_CI_lower': None if irr_ci_lower is None else float(irr_ci_lower),
        'IRR_CI_upper': None if irr_ci_upper is None else float(irr_ci_upper),
        'pvalue': None if pvalue is None else float(pvalue),
        'significant_at_0.05': bool(significant),
        'conclusion': conclusion
    }

    description = (
        "Extracted statistics for predictor 'SkinDark'. 'IRR' is the incidence rate ratio "
        "for red cards comparing dark-skinned (1) to light-skinned (0) players, adjusted for model covariates. "
        "'pvalue' is for the null hypothesis that the coefficient = 0 (IRR = 1). "
        "The 'conclusion' field gives a one-sentence interpretation in plain language."
    )

    return {"object": result_object, "description": description}