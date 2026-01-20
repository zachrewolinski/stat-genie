def extract_final_answer(model_output):
    """
    Extracts and interprets the Reader View effect and its interaction with dyslexia
    from a fitted statsmodels MixedLMResults (or wrapper).

    Returns a dict with keys:
      - "object": dict of numeric summaries (coefficients, SE, z, p, 95% CI, % change)
      - "description": concise interpretation of those results in the task context
    """
    import math

    res = model_output  # assumed to be a statsmodels MixedLMResultsWrapper or results object

    params = res.params            # pandas Series
    try:
        cov = res.cov_params()     # covariance matrix of params (DataFrame)
    except Exception:
        cov = res.cov_params()     # try again; if this fails the caller environment likely differs

    # Helper to find parameter name robustly
    def find_param_name(parts):
        # parts: list of substrings that must be present in the param name
        for name in params.index:
            if all(p in name for p in parts):
                return name
        return None

    # Expected parameter names (robust lookup)
    name_reader = find_param_name(['reader_view'])
    name_dys = find_param_name(['dyslexia_bin'])
    name_inter = find_param_name(['reader_view', 'dyslexia_bin', ':']) or find_param_name(['reader_view', 'dyslexia_bin'])

    # Prepare a small util to compute z,p,ci and percent change for a coeff and its se
    def summarize_coef(coef, se, alpha=0.05):
        z = coef / se if se != 0 else float('nan')
        # two-sided p-value from normal approx
        cdf = 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))
        p = 2 * (1 - cdf)
        z_crit = 1.96  # approximate for 95% CI
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se
        # percent change on original speed scale: exp(coef) - 1
        try:
            pct_change = math.exp(coef) - 1.0
        except OverflowError:
            pct_change = float('inf') if coef > 0 else float('-inf')
        return {
            'coef': float(coef),
            'se': float(se),
            'z': float(z),
            'p': float(p),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'pct_change': float(pct_change)
        }

    result_obj = {}
    # Extract main reader_view effect if present
    if name_reader is None:
        raise KeyError("Could not find a parameter corresponding to 'reader_view' in model params.")
    coef_reader = params[name_reader]
    se_reader = cov.loc[name_reader, name_reader] ** 0.5
    result_obj['reader_view_main'] = summarize_coef(coef_reader, se_reader)

    # Extract interaction effect (reader_view:dyslexia_bin) if present
    if name_inter is not None:
        coef_inter = params[name_inter]
        se_inter = cov.loc[name_inter, name_inter] ** 0.5
        result_obj['reader_view_x_dyslexia_interaction'] = summarize_coef(coef_inter, se_inter)
    else:
        coef_inter = 0.0
        # create a placeholder with NaNs if interaction absent
        result_obj['reader_view_x_dyslexia_interaction'] = {
            'coef': None, 'se': None, 'z': None, 'p': None,
            'ci_95_lower': None, 'ci_95_upper': None, 'pct_change': None
        }

    # Compute effect of Reader View for non-dyslexic (dyslexia_bin=0) and dyslexic (dyslexia_bin=1)
    # Non-dyslexic effect is simply the main coefficient for reader_view
    result_obj['effect_non_dyslexic'] = result_obj['reader_view_main']

    # Dyslexic effect = reader_view_main + interaction
    if name_inter is not None:
        coef_dys = coef_reader + coef_inter
        # Var(sum) = var(a) + var(b) + 2*cov(a,b)
        cov_ab = cov.loc[name_reader, name_inter]
        var_sum = cov.loc[name_reader, name_reader] + cov.loc[name_inter, name_inter] + 2.0 * cov_ab
        se_dys = var_sum ** 0.5 if var_sum >= 0 else float('nan')
        result_obj['effect_dyslexic'] = summarize_coef(coef_dys, se_dys)
    else:
        # If no interaction term, effect is same for both groups
        result_obj['effect_dyslexic'] = result_obj['reader_view_main']

    # Simple conclusion logic:
    # - We consider a two-sided alpha=0.05 cutoff on the dyslexic effect p-value
    dys_entry = result_obj['effect_dyslexic']
    non_dys_entry = result_obj['effect_non_dyslexic']

    if dys_entry['p'] is None:
        conclusion = ("Interaction term not present in model; the estimated Reader View effect "
                      "is the same for dyslexic and non-dyslexic participants. Interpret the "
                      "main effect accordingly.")
    else:
        # Interpret direction: negative coef => decrease in ln(ms) => decrease in time => faster reading
        if dys_entry['p'] < 0.05:
            if dys_entry['coef'] < 0:
                conclusion = ("There is statistically significant evidence (two-sided p = {:.3g}) that "
                              "Reader View changes reading time for dyslexic participants. The estimated "
                              "effect is a change of {:.3%} in the time measure (exp(coef)-1 = {:+.3%}), "
                              "i.e. a decrease in reading time (faster) since coef < 0.").format(
                                  dys_entry['p'], dys_entry['pct_change'], dys_entry['pct_change'])
            else:
                conclusion = ("There is statistically significant evidence (two-sided p = {:.3g}) that "
                              "Reader View increases reading time for dyslexic participants (slower), "
                              "with estimated change of {:.3%}.").format(
                                  dys_entry['p'], dys_entry['pct_change'])
        else:
            conclusion = ("No statistically significant effect of Reader View for dyslexic participants "
                          "(two-sided p = {:.3g}); estimated effect is {:+.3%} with 95% CI [{:+.3%}, {:+.3%}].").format(
                              dys_entry['p'],
                              dys_entry['pct_change'],
                              math.exp(dys_entry['ci_95_lower']) - 1.0,
                              math.exp(dys_entry['ci_95_upper']) - 1.0)

    # Prepare the returned dictionary
    return {
        "object": result_obj,
        "description": (
            "Extracted coefficients, SEs, z-statistics, p-values, 95% CI and percent-change (exp(coef)-1) "
            "for the Reader View main effect, its interaction with dyslexia, and the implied effects "
            "for dyslexic vs non-dyslexic participants. Conclusion: " + conclusion
        )
    }