def extract_final_answer(model_output):
    """
    Extracts key statistics from the model_output dictionary returned by the modeling function.
    Returns a dictionary with:
      - "object": a dict containing the dispersion, mean fish per hour (observed), model used,
                  and for each predictor: coefficient, IRR (exp(coef)), p-value, 95% CI (coef & IRR).
      - "description": a short textual summary interpreting the main results.
    """
    import numpy as np

    # Check required keys
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")
    for key in ('dispersion', 'descriptives'):
        if key not in model_output:
            raise ValueError(f"model_output missing required key: {key}")

    dispersion = model_output.get('dispersion', None)
    descriptives = model_output.get('descriptives', {})
    mean_fish_per_hour = descriptives.get('mean_fish_per_hour', None)

    # Prefer negative binomial if available (appropriate when dispersion >> 1)
    res = model_output.get('neg_binom') or model_output.get('poisson')
    if res is None:
        raise ValueError("No fitted model found in model_output under 'neg_binom' or 'poisson'.")

    params = res.params
    pvalues = getattr(res, 'pvalues', None)
    try:
        conf_int = res.conf_int()
    except Exception:
        # fallback: create approximate CI using coef +/- 1.96*se if conf_int() unavailable
        bse = getattr(res, 'bse', None)
        if bse is None:
            raise
        ci_lower = params - 1.96 * bse
        ci_upper = params + 1.96 * bse
        # assemble a 2-column structure similar to conf_int()
        conf_int = np.column_stack([ci_lower, ci_upper])
        # ensure indices align by converting to dict-like access later
        conf_int_indexed = True
    else:
        conf_int_indexed = hasattr(conf_int, 'loc')

    # predictors of interest
    predictors = ['livebait', 'camper', 'total_people_c']
    extracted = {'model_used': 'NegativeBinomial' if model_output.get('neg_binom') is not None else 'Poisson',
                 'dispersion': float(dispersion) if dispersion is not None else None,
                 'mean_fish_per_hour_observed': float(mean_fish_per_hour) if mean_fish_per_hour is not None else None,
                 'predictors': {}}

    summary_lines = []
    # note about model choice
    if dispersion is not None and dispersion > 1.5:
        summary_lines.append(f"Overdispersion detected (dispersion = {dispersion:.3f}); negative binomial model preferred.")
    else:
        summary_lines.append(f"Dispersion = {dispersion:.3f}; Poisson model may be adequate.")

    for name in predictors:
        if name not in params.index:
            # skip if predictor not in model (robustness)
            continue
        coef = float(params[name])
        irr = float(np.exp(coef))
        pval = float(pvalues[name]) if (pvalues is not None and name in pvalues.index) else None

        # extract conf int for coef
        try:
            if conf_int_indexed:
                ci_low, ci_high = conf_int.loc[name].values
            else:
                # conf_int is numpy array without index; find index of parameter
                idx = list(params.index).index(name)
                ci_low, ci_high = conf_int[idx, 0], conf_int[idx, 1]
        except Exception:
            # fallback to NaNs
            ci_low, ci_high = (np.nan, np.nan)
        irr_ci_low, irr_ci_high = (float(np.exp(ci_low)) if np.isfinite(ci_low) else None,
                                   float(np.exp(ci_high)) if np.isfinite(ci_high) else None)

        extracted['predictors'][name] = {
            'coef': coef,
            'coef_95ci': [float(ci_low) if np.isfinite(ci_low) else None,
                          float(ci_high) if np.isfinite(ci_high) else None],
            'IRR': irr,
            'IRR_95ci': [irr_ci_low, irr_ci_high],
            'pvalue': pval
        }

        signif = ("statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant")
        # interpretation phrasing differs for binary vs continuous predictor
        if name in ('livebait', 'camper'):
            interp = (f"Groups with {name} = 1 have an estimated rate that is {irr:.3f} times that of groups with {name} = 0 "
                      f"(95% CI for IRR: {irr_ci_low:.3f}–{irr_ci_high:.3f}; p = {pval:.3f} -> {signif}).")
        else:
            interp = (f"For a one-unit increase in {name} (this variable is mean-centered), the estimated rate multiplies by {irr:.3f} "
                      f"(95% CI for IRR: {irr_ci_low:.3f}–{irr_ci_high:.3f}; p = {pval:.3f} -> {signif}).")
        summary_lines.append(interp)

    # Add an overall observed mean rate line
    if mean_fish_per_hour is not None:
        summary_lines.insert(0, f"Observed mean fish caught per hour (raw): {mean_fish_per_hour:.3f} fish/hour.")

    description = " ".join(summary_lines)

    return {"object": extracted, "description": description}