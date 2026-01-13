def extract_final_answer(model_output):
    """
    Extract the effect of Reader View for readers with dyslexia from a fitted statsmodels OLS result.

    Returns a dictionary with:
      - "object": dict with numeric results (coef_sum, se, t, p, ci_lower, ci_upper, pct_change, pct_ci)
      - "description": human-readable interpretation of those results

    The function is robust to:
      - model_output being None (returns an explanatory message)
      - slightly different parameter naming (tries to locate the main and interaction terms)
      - missing covariance info (falls back to combining bse values if necessary)
    """
    import numpy as np
    try:
        from scipy import stats
    except Exception:
        stats = None

    # Check for None or invalid input
    if model_output is None:
        return {
            "object": None,
            "description": "model_output is None. No model results available to extract statistics."
        }

    # Try to get parameter series
    try:
        params = model_output.params  # pandas Series
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read params from model_output: {e}"
        }

    param_names = list(params.index.astype(str))

    # Heuristics to find main effect name and interaction name
    # Prefer exact 'reader_view' and 'reader_view:dyslexia_bin'
    main_name = None
    interaction_name = None

    # Exact matches first
    if 'reader_view' in param_names:
        main_name = 'reader_view'
    if 'reader_view:dyslexia_bin' in param_names:
        interaction_name = 'reader_view:dyslexia_bin'

    # If not found, try more flexible searches
    if main_name is None:
        # pick a param that contains 'reader_view' but not ':' (so it's the main term)
        for n in param_names:
            if ('reader_view' in n) and (':' not in n):
                main_name = n
                break

    if interaction_name is None:
        # pick a param that contains both 'reader_view' and 'dyslexia_bin' (likely interaction)
        for n in param_names:
            if ('reader_view' in n) and ('dyslexia_bin' in n):
                interaction_name = n
                break
        # also accept any param that contains both separated by ':'
        if interaction_name is None:
            for n in param_names:
                if ':' in n:
                    parts = n.split(':')
                    if any('reader_view' in p for p in parts) and any('dyslexia_bin' in p for p in parts):
                        interaction_name = n
                        break

    # If we have only the main effect (no interaction term found), then the effect for dyslexic readers is the main effect
    if main_name is None:
        return {
            "object": None,
            "description": "Could not locate a parameter corresponding to 'reader_view' in the model parameters."
        }

    if interaction_name is None:
        # Use main effect only
        coef_sum = float(params[main_name])
        # Try to get se for main_name
        try:
            bse_main = float(model_output.bse[main_name])
            se = bse_main
        except Exception:
            se = float('nan')

        # t, p, ci
        df = getattr(model_output, 'df_resid', np.nan)
        tstat = coef_sum / se if se and not np.isnan(se) else float('nan')
        pval = float('nan')
        ci_low = ci_high = float('nan')
        if (not np.isnan(se)) and (stats is not None) and (not np.isnan(df)):
            if df > 0:
                pval = stats.t.sf(abs(tstat), df) * 2
                tcrit = stats.t.ppf(1 - 0.025, df)
                ci_low = coef_sum - tcrit * se
                ci_high = coef_sum + tcrit * se

        pct_change = (np.exp(coef_sum) - 1) * 100 if not np.isnan(coef_sum) else float('nan')
        pct_ci = (np.exp(np.array([ci_low, ci_high])) - 1) * 100 if not np.isnan(ci_low) else [float('nan'), float('nan')]

        return {
            "object": {
                "coef_for_dyslexic_readers_log_wps": coef_sum,
                "se": se,
                "t": tstat,
                "p": pval,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "pct_change": pct_change,
                "pct_ci_lower": float(pct_ci[0]),
                "pct_ci_upper": float(pct_ci[1]),
                "notes": "No interaction term found; effect for dyslexic readers equals the main 'reader_view' coefficient."
            },
            "description": (
                "The model does not include a reader_view:dyslexia_bin interaction term, so the effect of Reader View "
                "for dyslexic readers is the main reader_view coefficient. Returned are the coefficient on the log(words/sec) scale, "
                "its standard error, t-statistic, p-value, 95% CI, and the percent change interpretation "
                "((exp(coef)-1)*100)."
            )
        }

    # Both main and interaction present -> compute combined effect for dyslexic readers
    try:
        coef_main = float(params[main_name])
        coef_int = float(params[interaction_name])
        coef_sum = coef_main + coef_int
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to read coefficient values for main or interaction term: {e}"
        }

    # Try to compute variance of the sum using covariance matrix
    se = None
    ci_low = ci_high = float('nan')
    pval = float('nan')
    tstat = float('nan')
    try:
        cov = model_output.cov_params()
        # cov may be a DataFrame; use .loc
        var_main = float(cov.loc[main_name, main_name])
        var_int = float(cov.loc[interaction_name, interaction_name])
        covar = float(cov.loc[main_name, interaction_name])
        var_sum = var_main + var_int + 2.0 * covar
        # Numerical safety
        if var_sum < 0 and var_sum > -1e-12:
            var_sum = 0.0
        se = float(np.sqrt(var_sum)) if var_sum >= 0 else float('nan')
    except Exception:
        # Fall back to adding bse in quadrature (ignores covariance)
        try:
            bse_main = float(model_output.bse[main_name])
            bse_int = float(model_output.bse[interaction_name])
            se = float(np.sqrt(bse_main ** 2 + bse_int ** 2))
        except Exception:
            se = float('nan')

    # Compute t, p, CI if possible
    df = getattr(model_output, 'df_resid', np.nan)
    if (not np.isnan(se)) and (se != 0):
        tstat = coef_sum / se
        if (stats is not None) and (not np.isnan(df)):
            try:
                pval = stats.t.sf(abs(tstat), df) * 2
                tcrit = stats.t.ppf(1 - 0.025, df)
                ci_low = coef_sum - tcrit * se
                ci_high = coef_sum + tcrit * se
            except Exception:
                pval = float('nan')
    else:
        tstat = float('nan')
        pval = float('nan')

    pct_change = (np.exp(coef_sum) - 1) * 100 if not np.isnan(coef_sum) else float('nan')
    pct_ci_low = (np.exp(ci_low) - 1) * 100 if not np.isnan(ci_low) else float('nan')
    pct_ci_high = (np.exp(ci_high) - 1) * 100 if not np.isnan(ci_high) else float('nan')

    description = (
        "Effect of Reader View for dyslexic readers equals (coef_reader_view + coef_reader_view:dyslexia_bin). "
        "Coefficients are on log(words-per-second); exp(coef)-1 gives proportional change. "
        "Returned: combined coefficient (log scale), its standard error, t-statistic, two-sided p-value, "
        "95% CI on the log scale and translated percent-change interpretation. "
        "A positive percent change means Reader View is associated with faster reading (higher words/sec) for dyslexic readers."
    )

    return {
        "object": {
            "main_param_name": main_name,
            "interaction_param_name": interaction_name,
            "coef_main": coef_main,
            "coef_interaction": coef_int,
            "coef_for_dyslexic_readers_log_wps": coef_sum,
            "se": se,
            "t": tstat,
            "p": pval,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "pct_change": pct_change,
            "pct_ci_lower": pct_ci_low,
            "pct_ci_upper": pct_ci_high
        },
        "description": description
    }