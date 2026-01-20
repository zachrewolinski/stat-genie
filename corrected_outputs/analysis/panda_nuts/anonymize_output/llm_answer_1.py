def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors
    of interest from a fitted statsmodels model (MixedLMResultsWrapper or OLS results).
    Returns a dictionary with keys:
      - "object": a dict containing model type, extracted effects for each predictor,
                  and random-intercept variance if available
      - "description": a short human-readable summary of each predictor's direction
                       and statistical significance in the context of nut-cracking efficiency.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Predictors of interest
    predictors = ['Age_c', 'Sex_male', 'Help', 'Age_c:Help']

    # Prepare containers
    effects = {}
    model_type = type(model_output).__name__

    # Try to get params, bse, pvalues, conf_int in a robust way
    # Different statsmodels result objects may expose different attributes
    try:
        params = model_output.params
    except Exception:
        # try attribute for mixed/fe params
        params = getattr(model_output, 'fe_params', None)

    try:
        bse = model_output.bse
    except Exception:
        bse = getattr(model_output, 'bse_fe', None)

    # p-values: if not provided, approximate using normal z from coef / se
    try:
        pvalues = model_output.pvalues
    except Exception:
        pvalues = None

    try:
        conf_int_df = model_output.conf_int()
    except Exception:
        conf_int_df = None

    # If pvalues missing and we have params & bse, compute approximate p-values
    if (pvalues is None or getattr(pvalues, 'empty', False)) and params is not None and bse is not None:
        z = params / bse
        pvalues = 2.0 * norm.sf(np.abs(z))
        # make pandas Series if params was Series
        if hasattr(params, 'index'):
            pvalues = pd.Series(pvalues, index=params.index)

    # If conf_int missing, compute approx using normal approximation
    if conf_int_df is None and params is not None and bse is not None:
        lower = params - 1.96 * bse
        upper = params + 1.96 * bse
        conf_int_df = pd.DataFrame({0: lower, 1: upper})

    # Extract random-intercept variance for MixedLM if available
    random_intercept_variance = None
    try:
        # MixedLM stores cov_re (covariance of random effects)
        cov_re = getattr(model_output, 'cov_re', None)
        if cov_re is not None:
            # cov_re might be a DataFrame or ndarray; take first diagonal element
            if hasattr(cov_re, 'iloc'):
                random_intercept_variance = float(cov_re.iloc[0, 0])
            else:
                # convert to numpy and take [0,0]
                cov_arr = np.asarray(cov_re)
                random_intercept_variance = float(cov_arr[0, 0])
        else:
            # some wrappers expose random effect var as "scale" or "vcomp" for simple models
            if hasattr(model_output, 'scale'):
                random_intercept_variance = float(model_output.scale)
    except Exception:
        random_intercept_variance = None

    # Fill effects dict for each predictor of interest
    for pred in predictors:
        if params is None or (not hasattr(params, '__contains__')) or (pred not in params.index):
            effects[pred] = None
            continue
        coef = float(params[pred])
        se = float(bse[pred]) if (bse is not None and pred in bse.index) else None
        pval = float(pvalues[pred]) if (pvalues is not None and pred in pvalues.index) else None

        # confidence interval: handle possible column names 0/1 or ['lower','upper']
        try:
            if 0 in conf_int_df.columns and 1 in conf_int_df.columns:
                ci_lower = float(conf_int_df.loc[pred, 0])
                ci_upper = float(conf_int_df.loc[pred, 1])
            else:
                # fallback to first two columns
                ci_lower = float(conf_int_df.iloc[conf_int_df.index.get_loc(pred), 0])
                ci_upper = float(conf_int_df.iloc[conf_int_df.index.get_loc(pred), 1])
        except Exception:
            # As a last resort compute approx if se available
            if se is not None:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = ci_upper = None

        significant = (pval is not None) and (pval < 0.05)

        effects[pred] = {
            'coef': coef,
            'se': se,
            'pvalue': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_at_0.05': bool(significant)
        }

    # Build a concise human-readable description
    desc_lines = []
    desc_lines.append(f"Model object type: {model_type}.")
    if random_intercept_variance is not None:
        desc_lines.append(f"Estimated random-intercept variance (ID): {random_intercept_variance:.4f}")
    else:
        desc_lines.append("Random-intercept variance: not available / not estimated.")

    for pred in predictors:
        info = effects.get(pred)
        if info is None:
            desc_lines.append(f"{pred}: term not present in the model output.")
            continue
        coef = info['coef']
        pval = info['pvalue']
        ci_l = info['ci_lower']
        ci_u = info['ci_upper']
        sign = "positive" if coef > 0 else ("no effect (≈0)" if np.isclose(coef, 0.0) else "negative")
        sig_text = f"statistically significant (p = {pval:.3g})" if info['significant_at_0.05'] else f"not statistically significant (p = {pval:.3g})"
        desc_lines.append(
            f"{pred}: {sign} effect on log(1 + nuts opened/min) (coef = {coef:.3f}); {sig_text}. "
            f"95% CI [{None if ci_l is None else format(ci_l, '.3f')}, {None if ci_u is None else format(ci_u, '.3f')}]."
        )

    description = " ".join(desc_lines)

    result_object = {
        'model_type': model_type,
        'effects': effects,
        'random_intercept_variance': random_intercept_variance
    }

    return {'object': result_object, 'description': description}