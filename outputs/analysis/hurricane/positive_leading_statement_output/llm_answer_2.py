def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, 95% CIs and interpret whether the results
    support the hypothesis that more feminine hurricane names are associated
    with greater harms (higher deaths).

    Input:
      model_output: dict of fitted statsmodels result objects, e.g.
        {'ols_masfem': <...>, 'ols_gender_mf': <...>, 'nb_masfem': <...>, ...}

    Returns:
      dict with keys:
        - "object": structured results (per-model stats + overall conclusion)
        - "description": short plain-English conclusion about the hypothesis
    """
    import math
    import numpy as np

    results_summary = {}
    # Map expected keys to the variable name of interest and a readable label
    model_map = {
        'ols_masfem': ('masfem', 'OLS on log deaths (masfem)'),
        'ols_gender_mf': ('gender_mf', 'OLS on log deaths (gender_mf)'),
        'nb_masfem': ('masfem', 'NegBin / GLM on counts (masfem)'),
        'nb_gender_mf': ('gender_mf', 'NegBin / GLM on counts (gender_mf)'),
    }

    def safe_get_conf_int(res, var):
        # res.conf_int() usually returns a DataFrame-like object indexed by param names
        try:
            ci = res.conf_int()
            # If conf_int has columns [0,1], try .loc
            if var in ci.index:
                lower, upper = float(ci.loc[var].iloc[0]), float(ci.loc[var].iloc[1])
            else:
                # If index is positional, try to find by matching order of params
                params = list(res.params.index) if hasattr(res, 'params') else None
                if params and var in params:
                    idx = params.index(var)
                    lower, upper = float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
                else:
                    lower, upper = float('nan'), float('nan')
            return lower, upper
        except Exception:
            return float('nan'), float('nan')

    # Iterate over models present in model_output and extract stats
    for key, (var, label) in model_map.items():
        if key not in model_output:
            continue
        res = model_output[key]
        model_info = {'label': label, 'variable': var}
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            if params is None or pvalues is None:
                raise AttributeError("Missing params or pvalues on result object")

            if var not in params.index:
                # variable not in model (e.g., due to collinearity or coding); skip
                model_info.update({
                    'present': False,
                    'note': f"Variable '{var}' not present in model parameters."
                })
                results_summary[key] = model_info
                continue

            coef = float(params[var])
            pval = float(pvalues[var])
            ci_lower, ci_upper = safe_get_conf_int(res, var)

            # Interpretation depending on model type
            if key.startswith('ols_'):
                # Outcome was log_alldeaths; interpret approximate percent change:
                # percent ≈ 100*(exp(coef)-1)
                try:
                    pct_change = (math.exp(coef) - 1.0) * 100.0
                except Exception:
                    pct_change = float('nan')
                interpretation = {
                    'scale': 'log outcome',
                    'interpretation': (
                        "Positive coef => higher log(deaths). "
                        "Approx percent change in deaths per unit increase = 100*(exp(coef)-1)."
                    ),
                    'approx_pct_change': pct_change
                }
            else:
                # GLM (NegBin/Poisson) with log link: exp(coef) is multiplicative effect on mean deaths
                try:
                    mult = math.exp(coef)
                except Exception:
                    mult = float('nan')
                interpretation = {
                    'scale': 'log link (count model)',
                    'interpretation': (
                        "Coef on log scale. exp(coef) = multiplicative change in expected deaths "
                        "per unit increase in predictor."
                    ),
                    'exp_coef': mult
                }

            significant = (pval < 0.05)
            supports_hypothesis = None
            # Hypothesis: more feminine names -> more harms (higher deaths).
            if coef > 0 and significant:
                supports_hypothesis = True
            elif coef < 0 and significant:
                supports_hypothesis = False
            else:
                supports_hypothesis = None  # inconclusive / not significant

            model_info.update({
                'present': True,
                'coef': coef,
                'p_value': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant_0.05': bool(significant),
                'supports_hypothesis': supports_hypothesis,
                'interpretation': interpretation
            })
        except Exception as e:
            model_info.update({
                'present': False,
                'error_extracting': str(e)
            })
        results_summary[key] = model_info

    # Aggregate a simple conclusion rule across primary models:
    # Consider primary evidence = ols_masfem and nb_masfem if present,
    # and use gender_mf models as secondary checks.
    primary_keys = [k for k in ['ols_masfem', 'nb_masfem'] if k in results_summary]
    secondary_keys = [k for k in ['ols_gender_mf', 'nb_gender_mf'] if k in results_summary]

    pos_sig = 0
    neg_sig = 0
    nonsig = 0
    checked_models = []

    for k in primary_keys + secondary_keys:
        info = results_summary.get(k)
        if not info or not info.get('present', False):
            continue
        checked_models.append(k)
        s = info.get('supports_hypothesis')
        if s is True:
            pos_sig += 1
        elif s is False:
            neg_sig += 1
        else:
            nonsig += 1

    # Formulate conclusion
    if len(checked_models) == 0:
        conclusion = "No relevant fitted models found in input to evaluate the hypothesis."
    else:
        if pos_sig >= 1 and neg_sig == 0:
            conclusion = (
                "Results provide evidence supporting the hypothesis: at least one model "
                "shows a statistically significant positive association between name femininity "
                "and harms (higher deaths)."
            )
        elif neg_sig >= 1 and pos_sig == 0:
            conclusion = (
                "Results provide evidence against the hypothesis: at least one model "
                "shows a statistically significant negative association (more feminine names associated "
                "with fewer deaths)."
            )
        elif pos_sig == 0 and neg_sig == 0:
            conclusion = (
                "No statistically significant association detected in the examined models. "
                "Evidence is inconclusive; coefficients may be positive or negative but are not statistically significant."
            )
        else:
            conclusion = (
                "Mixed evidence: some models show a statistically significant positive association "
                "and others show a statistically significant negative association. The overall evidence is inconclusive."
            )

    return {
        "object": {
            "per_model": results_summary,
            "checked_models": checked_models,
            "counts": {"positive_significant": pos_sig, "negative_significant": neg_sig, "non_significant": nonsig},
            "conclusion": conclusion
        },
        "description": (
            "This output contains per-model coefficient estimates, p-values, and 95% CIs for the "
            "focal predictors (masfem and gender_mf). For OLS on log(deaths) coefficients are approximately "
            "interpretable as percent changes = 100*(exp(coef)-1). For count GLMs with log link, exp(coef) "
            "is the multiplicative change in expected deaths. The 'conclusion' field gives a concise "
            "judgement about whether the fitted models support the hypothesis that more feminine names "
            "are associated with greater harms (higher deaths)."
        )
    }