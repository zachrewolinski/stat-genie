def extract_final_answer(model_output):
    """
    Extract key statistics for the primary predictors from a fitted statsmodels
    MixedLMResults (or MixedLMResultsWrapper) object.

    Returns a dictionary with:
      - "object": dict containing per-predictor statistics and model fit info
      - "description": brief interpretation of the extracted statistics

    The per-predictor dictionary includes:
      - coef: estimated coefficient on log-efficiency
      - se: standard error
      - z: test statistic (coef / se)
      - p: two-sided p-value
      - ci_2.5% / ci_97.5%: 95% confidence interval on the coefficient
      - exp_coef: exponentiated coefficient (multiplicative effect on nuts/min)
      - pct_change: (exp_coef - 1) * 100, approximate percent change in nuts/min
      - significant: boolean for p < 0.05
      - interpretation: short plain-language interpretation

    Notes:
      - For the log-transformed dependent variable, exp(coef) gives the multiplicative
        factor on the original nuts/min rate (e.g., exp(coef)=1.10 => ~10% increase).
      - The function expects predictors with names: 'Age_c', 'Sex_Male',
        'Help_binary', 'Age_c:Help_binary', 'Sex_Male:Help_binary'. If any are
        absent from the model, the entry will be None.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Pull parameter estimates and standard errors
    params = getattr(res, "params", None)
    bse = getattr(res, "bse", None)

    if params is None or bse is None:
        raise ValueError("model_output does not have params or bse attributes expected for statsmodels results.")

    # Compute z-statistics and p-values (use provided pvalues if available)
    z_vals = params / bse
    try:
        pvals = res.pvalues
    except Exception:
        pvals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_vals)))

    # Confidence intervals
    try:
        ci_df = res.conf_int(alpha=0.05)
    except Exception:
        # If conf_int not available, approximate with coef +/- 1.96*se
        ci_low = params - 1.96 * bse
        ci_high = params + 1.96 * bse
        ci_df = np.column_stack((ci_low, ci_high))
        # convert to a structure with index matching params
        import pandas as _pd
        ci_df = _pd.DataFrame(ci_df, index=params.index, columns=["2.5%", "97.5%"])

    # Predictors of interest
    predictors = ['Age_c', 'Sex_Male', 'Help_binary', 'Age_c:Help_binary', 'Sex_Male:Help_binary']

    results = {}
    for pred in predictors:
        if pred in params.index:
            coef = float(params[pred])
            se = float(bse[pred])
            zval = float(z_vals[pred])
            pval = float(pvals[pred]) if (hasattr(pvals, 'index') and pred in pvals.index) else float(2.0 * (1.0 - stats.norm.cdf(abs(zval))))
            # CI extraction: handle DataFrame or ndarray form
            try:
                ci_low = float(ci_df.loc[pred].iat[0])
                ci_high = float(ci_df.loc[pred].iat[1])
            except Exception:
                # fallback if ci_df is numpy array-like with same ordering
                try:
                    idx = list(params.index).index(pred)
                    ci_low = float(ci_df[idx, 0])
                    ci_high = float(ci_df[idx, 1])
                except Exception:
                    ci_low = None
                    ci_high = None

            exp_coef = float(np.exp(coef))
            pct_change = (exp_coef - 1.0) * 100.0
            significant = (pval < 0.05)

            # Plain-language interpretation
            if significant:
                if coef > 0:
                    interp = f"Statistically significant positive association: estimated {pct_change:.1f}% increase in nuts/min (exp(coef)={exp_coef:.3f})."
                else:
                    interp = f"Statistically significant negative association: estimated {abs(pct_change):.1f}% decrease in nuts/min (exp(coef)={exp_coef:.3f})."
            else:
                interp = "No strong evidence of an effect (p >= 0.05)."

            results[pred] = {
                'coef': coef,
                'se': se,
                'z': zval,
                'p': pval,
                'ci_2.5%': ci_low,
                'ci_97.5%': ci_high,
                'exp_coef': exp_coef,
                'pct_change': pct_change,
                'significant': bool(significant),
                'interpretation': interp
            }
        else:
            results[pred] = None

    # Add some model-level info if available
    model_info = {}
    if hasattr(res, 'nobs'):
        try:
            model_info['nobs'] = int(res.nobs)
        except Exception:
            model_info['nobs'] = float(res.nobs)
    if hasattr(res, 'model') and hasattr(res.model, 'groups'):
        try:
            model_info['n_groups'] = len(np.unique(res.model.groups))
        except Exception:
            pass
    for attr in ('aic', 'bic', 'llf'):
        if hasattr(res, attr):
            try:
                model_info[attr] = float(getattr(res, attr))
            except Exception:
                model_info[attr] = getattr(res, attr)

    description_lines = [
        "Extracted coefficients, standard errors, z-statistics, two-sided p-values,",
        "95% confidence intervals, exponentiated coefficients (multiplicative effect on nuts/min),",
        "and percent change for primary predictors: Age_c, Sex_Male, Help_binary,",
        "and their interactions (Age_c:Help_binary, Sex_Male:Help_binary).",
        "",
        "Interpretation guidance:",
        "- For the log-transformed outcome, exp(coef) gives the factor change in nuts/min.",
        "- Percent change = (exp(coef) - 1) * 100; positive = increase in efficiency, negative = decrease.",
        "- 'significant' flags predictors with p < 0.05."
    ]
    description = " ".join(description_lines)

    return {
        "object": {
            "predictors": results,
            "model_info": model_info
        },
        "description": description
    }