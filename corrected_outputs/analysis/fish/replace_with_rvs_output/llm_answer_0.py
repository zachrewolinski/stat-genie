def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted model object returned by the modeling function.

    Returns a dict with:
      - "object": a dict with numeric results (coefficients, SEs, p-values, 95% CIs,
                  incident-rate-ratios (IRRs) and their CIs, dispersion, and predicted
                  catch-rates per hour for livebait=0 vs livebait=1 holding camper=0
                  and group_size at its median).
      - "description": a short plain-language interpretation of the livebait effect
                       (and brief notes on camper and group_size).
    """
    import numpy as _np

    # Get chosen model (fall back to poisson if chosen_model missing)
    model = None
    if isinstance(model_output, dict):
        model = model_output.get('chosen_model') or model_output.get('poisson')
    else:
        model = model_output

    if model is None:
        raise ValueError("No fitted model found in model_output (expected key 'chosen_model' or 'poisson').")

    # Extract coefficient table
    params = model.params.copy()
    bse = model.bse.copy()
    pvalues = model.pvalues.copy()
    try:
        conf = model.conf_int()
    except Exception:
        # conf_int sometimes returns different structure; try to build from params +/- 1.96*bse
        conf = _np.vstack((params - 1.96 * bse, params + 1.96 * bse)).T
        # Make a simple index-compatible structure
        conf = {name: conf[i, :] for i, name in enumerate(params.index)}

    # Ensure conf is accessible as array with indices
    if hasattr(conf, "loc") or hasattr(conf, "iloc"):
        # statsmodels returns a DataFrame-like object
        conf_lower = _np.asarray(conf.iloc[:, 0]).astype(float)
        conf_upper = _np.asarray(conf.iloc[:, 1]).astype(float)
        conf_index = list(conf.index)
    else:
        # conf is dict or ndarray-like
        if isinstance(conf, dict):
            conf_index = list(conf.keys())
            conf_lower = _np.array([conf[k][0] for k in conf_index], dtype=float)
            conf_upper = _np.array([conf[k][1] for k in conf_index], dtype=float)
        else:
            # ndarray: assume rows correspond to params in order
            conf_arr = _np.asarray(conf, dtype=float)
            conf_lower = conf_arr[:, 0]
            conf_upper = conf_arr[:, 1]
            conf_index = list(params.index)

    # Build numeric summary
    coef_summary = {}
    for i, name in enumerate(params.index):
        coef_summary[name] = {
            "coef": float(params[name]),
            "se": float(bse[name]),
            "pvalue": float(pvalues[name]),
            "ci_lower": float(conf_lower[i]),
            "ci_upper": float(conf_upper[i]),
            "irr": float(_np.exp(params[name])),
            "irr_ci_lower": float(_np.exp(conf_lower[i])),
            "irr_ci_upper": float(_np.exp(conf_upper[i]))
        }

    # Dispersion reported by model_output (Poisson Pearson dispersion). Might be None.
    dispersion = model_output.get('dispersion') if isinstance(model_output, dict) else None

    # Predicted rate per hour for a 1-hour offset (offset = log(1) = 0) for two scenarios:
    # livebait=0 vs livebait=1, holding camper=0, group_size at median from model's exog
    # (fall back to 0 if group_size not available)
    exog_names = None
    try:
        exog_names = list(model.model.exog_names)
    except Exception:
        exog_names = None

    # Find index for group_size in model.exog
    median_group_size = 0.0
    try:
        if exog_names and 'group_size' in exog_names:
            idx = exog_names.index('group_size')
            exog_array = _np.asarray(model.model.exog)
            median_group_size = float(_np.median(exog_array[:, idx]))
        else:
            # Try to infer from params presence; if group_size param exists but exog_names missing,
            # we cannot compute median; leave at 0.
            if 'group_size' not in params.index:
                median_group_size = 0.0
    except Exception:
        median_group_size = 0.0

    # Helper to compute rate per hour given covariates (offset=1 hour)
    def _rate_per_hour(livebait_val, camper_val, group_size_val):
        linpred = 0.0
        # Add const if present
        if 'const' in params.index:
            linpred += params['const']
        # Add other terms if present
        if 'livebait' in params.index:
            linpred += params['livebait'] * livebait_val
        if 'camper' in params.index:
            linpred += params['camper'] * camper_val
        if 'group_size' in params.index:
            linpred += params['group_size'] * group_size_val
        return float(_np.exp(linpred))

    rate_no_livebait = _rate_per_hour(livebait_val=0, camper_val=0, group_size_val=median_group_size)
    rate_with_livebait = _rate_per_hour(livebait_val=1, camper_val=0, group_size_val=median_group_size)
    rate_ratio_empirical = rate_with_livebait / rate_no_livebait if rate_no_livebait > 0 else None

    # Compose the object to return
    obj = {
        "coef_table": coef_summary,
        "dispersion": float(dispersion) if (dispersion is not None) else None,
        "predicted_rate_per_hour": {
            "group_size_median_used": float(median_group_size),
            "camper": 0,
            "rate_no_livebait_per_hour": float(rate_no_livebait),
            "rate_with_livebait_per_hour": float(rate_with_livebait),
            "empirical_rate_ratio_livebait": float(rate_ratio_empirical) if rate_ratio_empirical is not None else None
        }
    }

    # Short interpretation focusing on livebait (primary independent variable),
    # plus brief notes on camper and group_size.
    if 'livebait' in coef_summary:
        c = coef_summary['livebait']
        interpretation = (
            f"The model chosen was {'Negative Binomial' if model.model.family.__class__.__name__.lower().find('negative')>=0 else 'Poisson'}; "
            f"Pearson dispersion reported = {obj['dispersion']:.3f}.\n"
            f"Livebait effect: coefficient = {c['coef']:.3f} (SE = {c['se']:.3f}, p = {c['pvalue']:.3g}),\n"
            f"  IRR = {c['irr']:.3f} with 95% CI [{c['irr_ci_lower']:.3f}, {c['irr_ci_upper']:.3f}].\n"
            f"  Interpretation: holding camper and group_size constant at the referenced values, groups using live bait are estimated to catch about {c['irr']:.2f} times as many fish per hour as groups not using live bait. "
            f"For a typical group (group_size median = {obj['predicted_rate_per_hour']['group_size_median_used']:.2f}, camper=0), "
            f"predicted catch-rate per hour = {obj['predicted_rate_per_hour']['rate_no_livebait_per_hour']:.3f} (no livebait) vs {obj['predicted_rate_per_hour']['rate_with_livebait_per_hour']:.3f} (with livebait).\n"
        )
    else:
        interpretation = "The model does not contain a 'livebait' coefficient; cannot interpret livebait effect."

    # Add brief notes on camper and group_size
    notes = []
    if 'camper' in coef_summary:
        cc = coef_summary['camper']
        notes.append(f"Camper: IRR = {cc['irr']:.3f} (95% CI [{cc['irr_ci_lower']:.3f}, {cc['irr_ci_upper']:.3f}], p = {cc['pvalue']:.3g}).")
    if 'group_size' in coef_summary:
        gg = coef_summary['group_size']
        notes.append(f"Group size: coefficient = {gg['coef']:.3f} (IRR per additional person = {gg['irr']:.3f}, p = {gg['pvalue']:.3g}).")

    if notes:
        interpretation += "Notes: " + " ".join(notes)

    return {"object": obj, "description": interpretation}