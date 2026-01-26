def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs and interpretable effect sizes
    for the key predictors (masfem_z and gender_mf) from the provided model_output dict.

    Returns a dictionary:
      {
        "object": {
           "ols_deaths": { "masfem_z": {coef, se, p_value, ci_95, percent_change, percent_change_CI_95}, "gender_mf": {...}, ...},
           "ols_damage": { ... },
           "nb_deaths": { "masfem_z": {coef, se, p_value, ci_95, IRR, IRR_CI_95, percent_change}, ...}
        },
        "description": "Plain-language explanation of what the numbers mean and how to interpret them re: the hypothesis"
      }

    Notes on interpretation encoded in results:
      - For OLS on log outcomes: percent_change = (exp(coef) - 1) * 100 gives the multiplicative percent change in the outcome
        associated with a one-unit increase in the predictor (masfem_z is standardized, so per SD).
      - For Negative Binomial: IRR = exp(coef) is the incidence-rate ratio; percent_change = (IRR - 1) * 100.
    """
    import numpy as np
    results_summary = {}

    # helper to safely extract attributes
    def safe_attr(res, attr_name):
        try:
            return getattr(res, attr_name)
        except Exception:
            return None

    for key in ['ols_deaths', 'ols_damage', 'nb_deaths']:
        res = model_output.get(key, None)
        results_summary[key] = {}

        if res is None:
            results_summary[key]['error'] = 'model not present (None)'
            continue

        if isinstance(res, Exception):
            # model fitting raised an exception earlier
            results_summary[key]['error'] = f'model object is an Exception: {str(res)}'
            continue

        # Try to extract params, bse, pvalues, conf_int
        params = safe_attr(res, 'params')
        pvalues = safe_attr(res, 'pvalues')
        bse = safe_attr(res, 'bse')
        try:
            conf = res.conf_int(alpha=0.05)
        except Exception:
            conf = None

        if params is None:
            results_summary[key]['error'] = 'could not extract params from model object'
            continue

        for var in ['masfem_z', 'gender_mf']:
            if var not in params.index:
                results_summary[key][var] = {'error': f'{var} not in model coefficients'}
                continue

            coef = float(params.loc[var])
            se = float(bse.loc[var]) if (bse is not None and var in bse.index) else None
            pval = float(pvalues.loc[var]) if (pvalues is not None and var in pvalues.index) else None
            if conf is not None and var in conf.index:
                ci_lower = float(conf.loc[var].iloc[0])
                ci_upper = float(conf.loc[var].iloc[1])
            else:
                ci_lower = ci_upper = None

            if key.startswith('ols'):
                # Outcome is log-transformed -> interpret multiplicatively
                # Use exp(coef)-1 for exact percent change
                try:
                    pct_change = (np.exp(coef) - 1.0) * 100.0
                    pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
                    pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
                except Exception:
                    pct_change = pct_ci_lower = pct_ci_upper = None

                results_summary[key][var] = {
                    'coef': coef,
                    'se': se,
                    'p_value': pval,
                    'ci_95': (ci_lower, ci_upper),
                    'percent_change': pct_change,
                    'percent_change_CI_95': (pct_ci_lower, pct_ci_upper),
                    'interpretation_note': (
                        "For log outcome: percent_change = (exp(coef)-1)*100 gives the percent change in the outcome "
                        "for a one-unit increase in the predictor. masfem_z is standardized, so this is per SD."
                    )
                }
            else:
                # Negative Binomial: coef on log scale -> IRR
                try:
                    irr = float(np.exp(coef))
                    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
                    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
                    pct_change = (irr - 1.0) * 100.0
                except Exception:
                    irr = irr_ci_lower = irr_ci_upper = pct_change = None

                results_summary[key][var] = {
                    'coef': coef,
                    'se': se,
                    'p_value': pval,
                    'ci_95': (ci_lower, ci_upper),
                    'IRR': irr,
                    'IRR_CI_95': (irr_ci_lower, irr_ci_upper),
                    'percent_change': pct_change,
                    'interpretation_note': (
                        "For Negative Binomial: IRR = exp(coef). IRR>1 means a higher expected count with higher predictor; "
                        "percent_change = (IRR-1)*100."
                    )
                }

        # Add quick verdict for masfem_z relative to hypothesis
        mm = results_summary[key].get('masfem_z')
        if isinstance(mm, dict) and 'coef' in mm:
            coef = mm['coef']
            pval = mm['p_value']
            supports_hypothesis = None
            # Hypothesis: more feminine -> more harm -> coef should be positive
            if pval is not None:
                supports_hypothesis = (coef > 0) and (pval < 0.05)
            else:
                supports_hypothesis = (coef > 0)
            results_summary[key]['masfem_z_summary'] = {
                'sign': 'positive' if coef > 0 else ('zero' if coef == 0 else 'negative'),
                'p_value': pval,
                'supports_hypothesis_at_p_lt_0.05': bool(supports_hypothesis)
            }

    description_lines = [
        "This function returns extracted statistics for 'masfem_z' (numeric femininity) and 'gender_mf' (binary) from each model in model_output.",
        "- For OLS models on log outcomes (ols_deaths, ols_damage): 'percent_change' reports (exp(coef)-1)*100, the multiplicative percent change in the outcome for a one-unit increase in the predictor (masfem_z is standardized, so interpreted per SD).",
        "- For the Negative Binomial model (nb_deaths): 'IRR' = exp(coef) and 'percent_change' = (IRR-1)*100 give the multiplicative change in expected death counts.",
        "- The returned 'masfem_z_summary' for each model gives a quick yes/no style check whether the estimate is positive and statistically significant at p < 0.05 (which would be consistent with the hypothesis that more feminine names lead to more harm).",
        "Use the numeric outputs in 'object' to make precise statements about direction, magnitude, and statistical significance."
    ]

    return {'object': results_summary, 'description': " ".join(description_lines)}