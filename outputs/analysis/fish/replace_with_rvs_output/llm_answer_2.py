def extract_final_answer(model_output):
    """
    Extract interpretable statistics from a statsmodels GLMResultsWrapper (Poisson or NB) that
    modeled fish counts with a log-hours offset.

    Returns a dict with:
      - "object": dict with numeric results (coefficients, IRRs, CIs, baseline rates, dispersion, predicted rates)
      - "description": brief textual interpretation of key results in context

    Expected model covariate names: 'const', 'livebait', 'camper', 'persons_total_centered'
    The function is robust to missing covariates (will omit or warn).
    """
    import numpy as np
    import pandas as pd

    out = {}
    try:
        res = model_output
        params = res.params.copy()           # pandas Series
        bse = res.bse.copy()                 # standard errors
        pvals = res.pvalues.copy()
        ci = res.conf_int().copy()           # DataFrame with [lower, upper]
        mu = res.predict()                   # fitted mean (on response scale for GLM)
        endog = res.model.endog
        df_resid = getattr(res, 'df_resid', None)
        model_family = getattr(res.model, 'family', None)
        family_name = type(model_family).__name__ if model_family is not None else 'Unknown'
    except Exception as e:
        raise ValueError(f"Provided object does not look like a fitted statsmodels GLMResultsWrapper: {e}")

    # Basic model info
    out['model_type'] = family_name

    # Compute Pearson dispersion (recompute here for clarity)
    eps = 1e-8
    try:
        pearson_chi2 = np.sum(((endog - mu) ** 2) / (mu + eps))
        if df_resid is None or df_resid <= 0:
            dispersion = np.nan
        else:
            dispersion = pearson_chi2 / df_resid
    except Exception:
        pearson_chi2 = np.nan
        dispersion = np.nan
    out['pearson_chi2'] = float(pearson_chi2) if np.isfinite(pearson_chi2) else None
    out['dispersion'] = float(dispersion) if np.isfinite(dispersion) else None

    # Prepare coefficient table with IRRs and CIs
    coeffs = {}
    for name in params.index:
        coef = float(params[name])
        se = float(bse.get(name, np.nan))
        p = float(pvals.get(name, np.nan))
        ci_lower, ci_upper = tuple(ci.loc[name]) if name in ci.index else (np.nan, np.nan)

        # For GLM with log link, exponentiated coefficients are incident rate ratios (IRR)
        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_lower)) if np.isfinite(ci_lower) else None
        irr_ci_upper = float(np.exp(ci_upper)) if np.isfinite(ci_upper) else None

        coeffs[name] = {
            'coef': coef,
            'std_err': se,
            'p_value': p,
            'ci_95': (float(ci_lower) if np.isfinite(ci_lower) else None,
                      float(ci_upper) if np.isfinite(ci_upper) else None),
            'irr': irr,
            'irr_95_ci': (irr_ci_lower, irr_ci_upper)
        }
    out['coefficients'] = coeffs

    # Baseline rate per hour: intercept (const) is log(rate per hour) when covariates = 0
    if 'const' in params.index:
        intercept = float(params['const'])
        intercept_ci = tuple(ci.loc['const']) if 'const' in ci.index else (np.nan, np.nan)
        baseline_rate = float(np.exp(intercept))
        baseline_rate_ci = (float(np.exp(intercept_ci[0])) if np.isfinite(intercept_ci[0]) else None,
                            float(np.exp(intercept_ci[1])) if np.isfinite(intercept_ci[1]) else None)
        out['baseline_rate_per_hour'] = baseline_rate
        out['baseline_rate_per_hour_95ci'] = baseline_rate_ci
    else:
        out['baseline_rate_per_hour'] = None
        out['baseline_rate_per_hour_95ci'] = (None, None)

    # Helper to compute predicted rate per hour for given covariate values (persons_total_centered is numeric)
    def predict_rate_per_hour(livebait=0, camper=0, persons_total_centered=0.0):
        # Build a vector aligned with params.index
        x = pd.Series(0.0, index=params.index)
        if 'const' in x.index:
            x['const'] = 1.0
        if 'livebait' in x.index:
            x['livebait'] = float(livebait)
        if 'camper' in x.index:
            x['camper'] = float(camper)
        if 'persons_total_centered' in x.index:
            x['persons_total_centered'] = float(persons_total_centered)
        linpred = float(np.dot(params.values, x.values))
        rate = float(np.exp(linpred))
        return rate

    # Predicted rates for common profiles at mean group size (persons_total_centered = 0)
    preds = {}
    preds['no_livebait_no_camper_mean_size'] = predict_rate_per_hour(livebait=0, camper=0, persons_total_centered=0)
    preds['livebait_only_mean_size'] = predict_rate_per_hour(livebait=1, camper=0, persons_total_centered=0)
    preds['camper_only_mean_size'] = predict_rate_per_hour(livebait=0, camper=1, persons_total_centered=0)
    preds['livebait_and_camper_mean_size'] = predict_rate_per_hour(livebait=1, camper=1, persons_total_centered=0)

    # Effect of one additional person (persons_total_centered): multiplicative change per person
    if 'persons_total_centered' in params.index:
        per_person_irr = float(np.exp(params['persons_total_centered']))
        per_person_ci = tuple(np.exp(ci.loc['persons_total_centered'])) if 'persons_total_centered' in ci.index else (None, None)
        out['per_person_irr'] = per_person_irr
        out['per_person_irr_95ci'] = (float(per_person_ci[0]) if np.isfinite(per_person_ci[0]) else None,
                                      float(per_person_ci[1]) if np.isfinite(per_person_ci[1]) else None)
    else:
        out['per_person_irr'] = None
        out['per_person_irr_95ci'] = (None, None)

    out['predicted_rates_per_hour'] = preds

    # Compose a brief textual description
    # We'll summarize baseline rate and key effects (livebait and camper) with p-values
    parts = []
    parts.append(f"Model family: {family_name}.")
    if out['dispersion'] is not None:
        parts.append(f"Pearson dispersion estimate = {out['dispersion']:.3f}.")
    if out['baseline_rate_per_hour'] is not None:
        parts.append(f"Estimated baseline fish catch rate = {out['baseline_rate_per_hour']:.3f} fish/hour "
                     f"(when livebait=0, camper=0, at mean group size). "
                     f"95% CI ~ [{out['baseline_rate_per_hour_95ci'][0]:.3f}, {out['baseline_rate_per_hour_95ci'][1]:.3f}].")

    # Livebait effect
    if 'livebait' in coeffs:
        lr = coeffs['livebait']
        parts.append(
            f"Using live bait: IRR = {lr['irr']:.3f} (95% CI [{lr['irr_95_ci'][0]:.3f}, {lr['irr_95_ci'][1]:.3f}]), "
            f"p = {lr['p_value']:.3g}. Interpretation: groups using live bait catch about {lr['irr']:.2f}x as many fish per hour "
            f"as otherwise, holding other covariates constant."
        )
    else:
        parts.append("Livebait coefficient not present in model results.")

    # Camper effect
    if 'camper' in coeffs:
        cr = coeffs['camper']
        parts.append(
            f"Camper present: IRR = {cr['irr']:.3f} (95% CI [{cr['irr_95_ci'][0]:.3f}, {cr['irr_95_ci'][1]:.3f}]), "
            f"p = {cr['p_value']:.3g}. Interpretation: groups with a camper catch about {cr['irr']:.2f}x as many fish per hour, "
            f"all else equal."
        )
    else:
        parts.append("Camper coefficient not present in model results.")

    # Persons effect
    if out['per_person_irr'] is not None:
        parts.append(
            f"Per additional person (relative to mean group size): IRR = {out['per_person_irr']:.3f} "
            f"(95% CI [{out['per_person_irr_95ci'][0]:.3f}, {out['per_person_irr_95ci'][1]:.3f}]). "
            f"Interpretation: each additional person multiplies the catch rate by ~{out['per_person_irr']:.2f}."
        )
    else:
        parts.append("persons_total_centered coefficient not present in model results.")

    # Provide concrete predicted rates
    parts.append("Example predicted rates per hour (at mean group size): "
                 f"no livebait & no camper = {preds['no_livebait_no_camper_mean_size']:.3f}, "
                 f"livebait only = {preds['livebait_only_mean_size']:.3f}, "
                 f"camper only = {preds['camper_only_mean_size']:.3f}, "
                 f"both = {preds['livebait_and_camper_mean_size']:.3f}.")

    description = " ".join(parts)

    return {"object": out, "description": description}