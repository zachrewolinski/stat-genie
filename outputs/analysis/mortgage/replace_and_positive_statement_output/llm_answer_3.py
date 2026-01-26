def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of the 'female' indicator on mortgage acceptance
    from the `model_output` produced by the modeling function.

    Returns a dictionary with keys:
      - "object": a dict with numeric results and a short conclusion
      - "description": a brief interpretation of those numbers in plain English
    """
    import numpy as np
    from scipy.stats import norm

    # Validate input
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dictionary as returned by the modeling function.")

    res_wrapper = model_output.get('logit_robust_result')
    if res_wrapper is None:
        raise KeyError("model_output missing 'logit_robust_result' entry.")

    # Get parameter Series
    params = getattr(res_wrapper, 'params', None)
    if params is None:
        # Try to access underlying result
        params = getattr(getattr(res_wrapper, '_res', None), 'params', None)
    if params is None:
        raise RuntimeError("Could not find parameter estimates in the model result.")

    if 'female' not in params.index:
        raise KeyError("No 'female' coefficient found in model parameters.")

    # Extract coefficient
    coef = float(params['female'])

    # Try to get robust standard error
    se = None
    # 1) wrapper exposes bse_robust (numpy array aligned with params)
    bse_robust = getattr(res_wrapper, 'bse_robust', None)
    try:
        if bse_robust is not None:
            idx = list(params.index).index('female')
            se = float(np.asarray(bse_robust)[idx])
    except Exception:
        se = None

    # 2) fallback: use wrapper.cov_robust if available
    if se is None:
        cov_robust = getattr(res_wrapper, 'cov_robust', None)
        if cov_robust is not None:
            try:
                idx = list(params.index).index('female')
                se = float(np.sqrt(np.asarray(cov_robust).diagonal()[idx]))
            except Exception:
                se = None

    # 3) final fallback: use model-based bse from original result
    if se is None:
        try:
            se = float(getattr(res_wrapper._res, 'bse')[list(params.index).index('female')])
        except Exception:
            se = None

    # Compute z, p-value, and 95% CI for coefficient (using the selected se)
    if se is not None and se > 0:
        z = coef / se
        p_value = 2 * (1 - norm.cdf(abs(z)))
        ci_low = coef - norm.ppf(0.975) * se
        ci_high = coef + norm.ppf(0.975) * se
    else:
        z = None
        p_value = None
        ci_low = None
        ci_high = None

    # Try to obtain average marginal effect (AME) for 'female' using the fitted result's get_margeff
    ame = None
    ame_se = None
    ame_p = None
    ame_ci_low = None
    ame_ci_high = None
    try:
        margeff_res = res_wrapper.get_margeff(at='overall', method='dydx')
        # Prefer summary_frame() when available (returns a DataFrame)
        try:
            sf = margeff_res.summary_frame()
            # Attempt to find a row corresponding to 'female'
            if 'female' in sf.index:
                row = sf.loc['female']
            else:
                # If index does not label rows by variable name, match by parameter order
                pos = list(params.index).index('female')
                row = sf.iloc[pos]
            # Extract columns robust to slightly different column names
            # Common column names: 'dy/dx', 'Std. Err.', 'P>|z|', '[0.025', '0.975]'
            if 'dy/dx' in row.index:
                ame = float(row['dy/dx'])
            elif 'dy/dx' in sf.columns:
                ame = float(row['dy/dx'])
            else:
                # fallback to first numeric column
                ame = float(row.iloc[0])
            # Std. Err.
            if 'Std. Err.' in row.index:
                ame_se = float(row['Std. Err.'])
            elif 'Std. Err.' in sf.columns:
                ame_se = float(row['Std. Err.'])
            else:
                # fallback to second numeric column if present
                try:
                    ame_se = float(row.iloc[1])
                except Exception:
                    ame_se = None
            # p-value
            if 'P>|z|' in row.index:
                ame_p = float(row['P>|z|'])
            elif 'P>|z|' in sf.columns:
                ame_p = float(row['P>|z|'])
            else:
                # compute from ame and ame_se if available
                if ame is not None and ame_se is not None and ame_se > 0:
                    z_ame = ame / ame_se
                    ame_p = 2 * (1 - norm.cdf(abs(z_ame)))
            # CI columns (try common names)
            if '[0.025' in row.index and '0.975]' in row.index:
                ame_ci_low = float(row['[0.025'])
                ame_ci_high = float(row['0.975]'])
            else:
                # fallback compute from ame and ame_se
                if ame is not None and ame_se is not None:
                    ame_ci_low = ame - norm.ppf(0.975) * ame_se
                    ame_ci_high = ame + norm.ppf(0.975) * ame_se
        except Exception:
            # If summary_frame isn't available, try attributes on margeff_res
            if hasattr(margeff_res, 'margeff'):
                pos = list(params.index).index('female')
                try:
                    ame = float(np.asarray(margeff_res.margeff)[pos])
                except Exception:
                    ame = None
            if hasattr(margeff_res, 'margeff_se'):
                try:
                    ame_se = float(np.asarray(margeff_res.margeff_se)[pos])
                except Exception:
                    ame_se = None
            if ame is not None and ame_se is not None and ame_se > 0:
                z_ame = ame / ame_se
                ame_p = 2 * (1 - norm.cdf(abs(z_ame)))
                ame_ci_low = ame - norm.ppf(0.975) * ame_se
                ame_ci_high = ame + norm.ppf(0.975) * ame_se
    except Exception:
        # If anything fails, leave AME fields as None
        pass

    # Build a concise conclusion based on AME (preferred) or coefficient if AME not available.
    significance_level = 0.05
    conclusion = {}
    if ame is not None and ame_p is not None:
        sig = (ame_p < significance_level)
        direction = "higher" if ame > 0 else "lower" if ame < 0 else "no difference"
        conclusion_text = (
            f"Average marginal effect for being female = {ame:.4f} (SE={ame_se:.4f}, "
            f"p={ame_p:.3f}). Interpretation: being female is associated with a {abs(ame*100):.2f} "
            f"percentage-point {direction} probability of mortgage approval. "
            f"{'Statistically significant.' if sig else 'Not statistically significant.'}"
        )
        conclusion['basis'] = 'AME'
        conclusion['significant'] = sig
    elif p_value is not None:
        sig = (p_value < significance_level)
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        conclusion_text = (
            f"Logit coefficient for female = {coef:.4f} (robust SE={se:.4f}, z={z:.3f}, p={p_value:.3f}). "
            f"Sign indicates being female is associated with a {direction} log-odds of approval. "
            f"{'Statistically significant.' if sig else 'Not statistically significant.'}"
        )
        conclusion['basis'] = 'coef'
        conclusion['significant'] = sig
    else:
        conclusion_text = "Could not compute a robust statistical conclusion for the 'female' effect."
        conclusion['basis'] = None
        conclusion['significant'] = None

    # Assemble the object to return (numbers + conclusion)
    result_object = {
        'coef_female': coef,
        'coef_robust_se': se,
        'coef_z': z,
        'coef_p_value': p_value,
        'coef_95ci': (ci_low, ci_high),
        'ame_female': ame,
        'ame_se': ame_se,
        'ame_p_value': ame_p,
        'ame_95ci': (ame_ci_low, ame_ci_high),
        'conclusion_text': conclusion_text,
        'conclusion_meta': conclusion
    }

    # Short human-readable description
    description = (
        "Extracted the logit coefficient and (preferably) the average marginal effect for the "
        "'female' indicator. The AME, when available, gives the estimated change in probability "
        "of mortgage acceptance associated with being female (in probability points). The "
        "returned values include robust standard errors, p-values, 95% CIs, and a short conclusion "
        "about statistical significance."
    )

    return {"object": result_object, "description": description}