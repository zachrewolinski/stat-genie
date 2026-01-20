def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors
    of interest (age, sex, HelpReceived) from a statsmodels fitted model object.
    Works for MixedLMResultsWrapper, RegressionResultsWrapper (OLS), and robust
    cov results wrappers.

    Returns:
        dict with keys:
          - "object": dict mapping predictor names ("age", "sex", "HelpReceived")
                      to extracted statistics (coef, se, pvalue, 95% CI, percent change,
                      and significance at alpha=0.05).
          - "description": human-readable interpretation of these results in the
                           context of the task (effect on log-efficiency and approximate
                           percent change in nut-cracking efficiency).
    """
    import numpy as np
    import pandas as pd

    # Helper to raise a clear error if model_output is not as expected
    if model_output is None:
        raise ValueError("model_output is None. Provide a fitted statsmodels result object.")

    # Try to extract params, bse, pvalues, conf_int
    try:
        params = model_output.params
    except Exception:
        # Some wrappers expose .params as attribute; if not, try calling attribute
        try:
            params = getattr(model_output, 'params')
        except Exception as e:
            raise ValueError(f"Could not extract params from model_output: {e}")

    # Ensure params is a pandas Series for indexing
    if not isinstance(params, (pd.Series, pd.DataFrame)):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Unexpected format for params in model_output.")

    # bse and pvalues may or may not exist depending on wrapper type
    bse = getattr(model_output, 'bse', None)
    pvalues = getattr(model_output, 'pvalues', None)
    try:
        conf = model_output.conf_int()
    except Exception:
        # Try alternative method: some results return array
        try:
            ci = model_output.conf_int
            conf = ci()
        except Exception:
            conf = None

    # Convert to pandas structures where possible
    if isinstance(bse, (list, tuple, np.ndarray)):
        bse = pd.Series(bse, index=params.index)
    if isinstance(pvalues, (list, tuple, np.ndarray)):
        pvalues = pd.Series(pvalues, index=params.index)
    if isinstance(conf, (list, tuple, np.ndarray)):
        # assume shape (n_params, 2)
        conf = pd.DataFrame(conf, index=params.index, columns=['lower', 'upper'])
    elif isinstance(conf, pd.DataFrame):
        # standardize columns to ['lower','upper'] if needed
        if conf.shape[1] >= 2:
            conf = conf.iloc[:, :2].copy()
            conf.columns = ['lower', 'upper']

    # Define target parameter name patterns and friendly names
    # We expect parameter names containing 'age', containing 'sex' (C(sex)...), and 'HelpReceived'
    param_index = list(params.index.astype(str))
    def find_param(patterns):
        for p in param_index:
            for pat in patterns:
                if pat in p:
                    return p
        return None

    age_name = find_param(['age'])
    help_name = find_param(['HelpReceived', 'Help_Received', 'Help'])
    sex_name = find_param(['C(sex)', 'sex', 'C(sex)[T.', 'sex[T.'])  # broad matching

    # Collect results for each
    results = {}
    alpha = 0.05

    for key, pname in [('age', age_name), ('sex', sex_name), ('HelpReceived', help_name)]:
        if pname is None:
            # Parameter not present (e.g., sex may be reference level only)
            results[key] = {
                'present': False,
                'message': f"No parameter found for {key} in model parameters."
            }
            continue

        coef = float(params[pname])
        se = float(bse[pname]) if bse is not None and pname in bse.index else None
        pval = float(pvalues[pname]) if pvalues is not None and pname in pvalues.index else None
        if conf is not None and pname in conf.index:
            ci_lower = float(conf.loc[pname, 'lower'])
            ci_upper = float(conf.loc[pname, 'upper'])
        else:
            ci_lower = ci_upper = None

        # Because DV is log(nuts/sec), exponentiate coefficient to get multiplicative change
        try:
            pct_change = (np.exp(coef) - 1.0) * 100.0
            pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
            pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
        except Exception:
            pct_change = pct_ci_lower = pct_ci_upper = None

        significant = (pval is not None) and (pval < alpha)

        results[key] = {
            'present': True,
            'param_name': pname,
            'coef': coef,
            'se': se,
            'pvalue': pval,
            '95%_CI_coef': (ci_lower, ci_upper) if (ci_lower is not None and ci_upper is not None) else None,
            'percent_change': pct_change,                     # approximate percent change in nuts/sec
            '95%_CI_percent_change': (pct_ci_lower, pct_ci_upper) if (pct_ci_lower is not None and pct_ci_upper is not None) else None,
            'significant_at_0.05': bool(significant)
        }

    # Form a short textual interpretation
    interp_lines = []
    # Age
    if results['age']['present']:
        r = results['age']
        line = f"Age: coef={r['coef']:.4f}"
        if r['pvalue'] is not None:
            line += f", p={r['pvalue']:.3g}"
        if r['percent_change'] is not None:
            line += f" -> ~{r['percent_change']:.1f}% change per year"
            if r['95%_CI_percent_change'] is not None:
                lo, hi = r['95%_CI_percent_change']
                line += f" (95% CI {lo:.1f}% to {hi:.1f}%)"
        if r['significant_at_0.05']:
            line += " — statistically significant (α=0.05)."
        else:
            line += " — not statistically significant (α=0.05)."
        interp_lines.append(line)
    else:
        interp_lines.append("Age: parameter not found in model output.")

    # Sex
    if results['sex']['present']:
        r = results['sex']
        # Determine which sex the coefficient refers to from param_name
        pname = r['param_name']
        # Try to infer direction: typically C(sex)[T.m] means effect of being male vs reference female
        ref_info = ''
        if 'T.' in pname:
            # extract token after T.
            try:
                level = pname.split('T.')[1].strip(']').split(']')[0]
                ref_info = f" (effect of {level} vs reference level)"
            except Exception:
                ref_info = ''
        line = f"Sex ({pname}) : coef={r['coef']:.4f}"
        if r['pvalue'] is not None:
            line += f", p={r['pvalue']:.3g}"
        if r['percent_change'] is not None:
            line += f" -> ~{r['percent_change']:.1f}% change {ref_info}"
            if r['95%_CI_percent_change'] is not None:
                lo, hi = r['95%_CI_percent_change']
                line += f" (95% CI {lo:.1f}% to {hi:.1f}%)"
        if r['significant_at_0.05']:
            line += " — statistically significant (α=0.05)."
        else:
            line += " — not statistically significant (α=0.05)."
        interp_lines.append(line)
    else:
        interp_lines.append("Sex: parameter not found in model output (may be absorbed as reference).")

    # HelpReceived
    if results['HelpReceived']['present']:
        r = results['HelpReceived']
        line = f"HelpReceived: coef={r['coef']:.4f}"
        if r['pvalue'] is not None:
            line += f", p={r['pvalue']:.3g}"
        if r['percent_change'] is not None:
            line += f" -> ~{r['percent_change']:.1f}% change when help received vs not"
            if r['95%_CI_percent_change'] is not None:
                lo, hi = r['95%_CI_percent_change']
                line += f" (95% CI {lo:.1f}% to {hi:.1f}%)"
        if r['significant_at_0.05']:
            line += " — statistically significant (α=0.05)."
        else:
            line += " — not statistically significant (α=0.05)."
        interp_lines.append(line)
    else:
        interp_lines.append("HelpReceived: parameter not found in model output.")

    description = " | ".join(interp_lines)

    return {"object": results, "description": description}