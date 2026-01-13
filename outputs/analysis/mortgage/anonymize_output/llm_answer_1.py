def extract_final_answer(model_output):
    """
    Extracts the estimated effect of being female on mortgage approval from a fitted
    statsmodels GLM/GLMResultsWrapper (with robust cov results if used).
    
    Returns a dictionary with:
      - "object": a dict containing coefficients, robust SEs, p-values, 95% CIs,
                  odds ratios and odds-ratio CIs for:
                    * Female (effect for non-Black applicants)
                    * Female + Female_Black (effect for Black applicants)
      - "description": short explanation of what the returned numbers mean.
    
    The function is defensive about the exact type of model_output (plain results
    or robustcov results) and about whether .conf_int() returns a DataFrame/ndarray.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # fallback: use normal approximation via numpy
        class _Norm:
            @staticmethod
            def cdf(x): 
                return 0.5*(1 + np.erf(x/np.sqrt(2)))
        norm = _Norm()

    # Get parameter estimates, standard errors, p-values
    try:
        params = model_output.params.copy()
    except Exception as e:
        raise ValueError("model_output has no .params attribute") from e

    try:
        bse = model_output.bse.copy()
    except Exception:
        # If bse not present, attempt to compute from cov_params
        cov = None
        try:
            cov = model_output.cov_params()
        except Exception:
            raise ValueError("model_output has no .bse or .cov_params()") 
        # build bse from diagonal
        import pandas as pd
        if isinstance(cov, (pd.DataFrame,)):
            bse = np.sqrt(np.diag(cov.values))
            bse = pd.Series(bse, index=params.index)
        else:
            bse = np.sqrt(np.diag(cov))
            import pandas as pd
            bse = pd.Series(bse, index=params.index)

    try:
        pvalues = model_output.pvalues.copy()
    except Exception:
        # try compute from params/bse using normal approx
        z = params / bse
        pvalues = 2 * (1 - norm.cdf(np.abs(z)))
        import pandas as pd
        pvalues = pd.Series(pvalues, index=params.index)

    # confidence intervals (95%) from model_output.conf_int() when possible
    try:
        ci_raw = model_output.conf_int()
        # conf_int may be DataFrame or ndarray. Normalize to DataFrame with index=params.index
        import pandas as pd
        if isinstance(ci_raw, pd.DataFrame):
            ci_df = ci_raw.copy()
            # ensure columns are [lower, upper]
            ci_df.columns = ['2.5%', '97.5%']
        else:
            # assume ndarray with shape (k,2)
            ci_df = pd.DataFrame(ci_raw, index=params.index, columns=['2.5%', '97.5%'])
    except Exception:
        # fallback: build from params +/- 1.96*bse
        import pandas as pd
        lower = params - 1.96 * bse
        upper = params + 1.96 * bse
        ci_df = pd.DataFrame({'2.5%': lower, '97.5%': upper})

    # Covariance matrix needed to compute combined effect (Female + Female_Black)
    try:
        cov = model_output.cov_params()
        # convert to ndarray and ensure we can index by param names
        import pandas as pd
        if isinstance(cov, pd.DataFrame):
            cov_df = cov.copy()
        else:
            # cov is ndarray, turn into DataFrame using params.index
            cov_df = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        # as a last resort, build diagonal covariance from bse^2 (no covariances)
        import pandas as pd
        cov_df = pd.DataFrame(np.diag(bse.values**2), index=params.index, columns=params.index)

    # Helper to safely get param values
    def get_param(name):
        if name not in params.index:
            raise KeyError(f"Parameter '{name}' not found in model parameters: {list(params.index)}")
        return params.loc[name], bse.loc[name], pvalues.loc[name], ci_df.loc[name, '2.5%'], ci_df.loc[name, '97.5%']

    results = {}

    # Extract Female effect (this is the effect for the baseline race group, i.e., non-Black if Black is coded as 1)
    try:
        coef_f, se_f, p_f, ci_lo_f, ci_hi_f = get_param('Female')
    except KeyError as e:
        raise

    or_f = float(np.exp(coef_f))
    or_ci_f = (float(np.exp(ci_lo_f)), float(np.exp(ci_hi_f)))

    results['Female_nonBlack'] = {
        'coef_log_odds': float(coef_f),
        'se': float(se_f),
        'p_value': float(p_f),
        'ci_95_log_odds': [float(ci_lo_f), float(ci_hi_f)],
        'odds_ratio': or_f,
        'ci_95_odds_ratio': [or_ci_f[0], or_ci_f[1]],
        'interpretation': (
            "This coefficient is the log-odds difference of approval for female vs male "
            "applicants among the reference race group (Black=0). Odds ratio = exp(coef)."
        )
    }

    # Compute effect of Female for Black applicants: Female + Female_Black
    if 'Female_Black' in params.index:
        coef_fb = params.loc['Female'] + params.loc['Female_Black']
        # variance of sum = var(Female) + var(Female_Black) + 2*cov(Female, Female_Black)
        var_f = cov_df.loc['Female', 'Female']
        var_fb = cov_df.loc['Female_Black', 'Female_Black']
        cov_f_fb = cov_df.loc['Female', 'Female_Black']
        se_fb = sqrt(var_f + var_fb + 2 * cov_f_fb)
        # 95% CI for combined effect
        ci_lo_fb = coef_fb - 1.96 * se_fb
        ci_hi_fb = coef_fb + 1.96 * se_fb
        # p-value via normal approximation
        z_fb = coef_fb / se_fb if se_fb > 0 else np.nan
        p_fb = float(2 * (1 - norm.cdf(abs(z_fb)))) if not np.isnan(z_fb) else np.nan
        or_fb = float(np.exp(coef_fb))
        or_ci_fb = (float(np.exp(ci_lo_fb)), float(np.exp(ci_hi_fb)))

        results['Female_ifBlack'] = {
            'coef_log_odds': float(coef_fb),
            'se': float(se_fb),
            'p_value': float(p_fb),
            'ci_95_log_odds': [float(ci_lo_fb), float(ci_hi_fb)],
            'odds_ratio': or_fb,
            'ci_95_odds_ratio': [or_ci_fb[0], or_ci_fb[1]],
            'interpretation': (
                "This is the log-odds effect of being female (vs male) for Black applicants: "
                "coef(Female) + coef(Female_Black). Odds ratio = exp(coef)."
            )
        }
    else:
        results['Female_ifBlack'] = None

    # Also include raw parameter summary for Female and Female_Black (if present)
    summary_params = {}
    for name in ['Female', 'Female_Black']:
        if name in params.index:
            summary_params[name] = {
                'coef_log_odds': float(params.loc[name]),
                'se': float(bse.loc[name]),
                'p_value': float(pvalues.loc[name]),
                'ci_95_log_odds': [float(ci_df.loc[name, '2.5%']), float(ci_df.loc[name, '97.5%'])],
                'odds_ratio': float(np.exp(params.loc[name])),
                'ci_95_odds_ratio': [float(np.exp(ci_df.loc[name, '2.5%'])),
                                     float(np.exp(ci_df.loc[name, '97.5%']))]
            }
        else:
            summary_params[name] = None

    # Build description
    desc = (
        "Returned statistics are from the fitted logistic regression predicting mortgage approval. "
        "Key quantities:\n"
        "- 'Female_nonBlack': estimated effect of being female (vs male) among non-Black applicants (Black=0). "
        "Values shown: log-odds coef, robust SE, p-value, 95% CI (log-odds), odds ratio (exp(coef)), and OR CI.\n"
        "- 'Female_ifBlack': estimated effect of being female (vs male) among Black applicants (computed as Female + Female_Black), "
        "with its SE, p-value (normal approx), 95% CI, and OR. If Female_Black is absent, this will be None.\n"
        "All p-values and SEs use the model object's reported robust covariance if available (cov_params / .bse).\n"
        "Use the odds ratios to interpret multiplicative change in odds of approval for females vs males."
    )

    return {"object": {"summary_params": summary_params, "effects": results}, "description": desc}