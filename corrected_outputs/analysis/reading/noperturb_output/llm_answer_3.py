def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    simple (group-specific) effects for the Reader View treatment from a fitted
    statsmodels model object (MixedLMResults, RegressionResultsWrapper, or similar).

    Returns a dict with:
      - "object": dict with numeric results:
          * coef_reader_view: estimated coefficient for reader_view (log-speed)
          * p_reader_view: p-value for that coefficient (if available/derivable)
          * ci_reader_view: 95% CI (lower, upper)
          * coef_interaction: estimated coefficient for reader_view:dyslexia_bin (if present)
          * p_interaction: p-value for the interaction
          * ci_interaction: 95% CI for the interaction
          * effect_non_dyslexic: simple effect of Reader View for dyslexia_bin=0 (coef, se, p, 95%CI, pct_change)
          * effect_dyslexic: simple effect of Reader View for dyslexia_bin=1 (coef, se, p, 95%CI, pct_change)
            (pct_change = (exp(coef) - 1) * 100 on the speed+1 scale)
      - "description": short explanation of what the numbers mean and how to interpret them.
    """
    import math
    import numpy as np

    res = model_output

    # Helper: safe access to params, cov, bse, pvalues, conf_int
    try:
        params = res.params.copy()
    except Exception:
        raise ValueError("Model output does not expose .params")

    # covariance matrix
    try:
        cov = res.cov_params()
    except Exception:
        # try attribute name difference
        cov = None

    # bse
    try:
        bse = res.bse
    except Exception:
        if cov is not None:
            bse = np.sqrt(np.diag(cov))
            bse = (params.index.to_list(), bse)  # placeholder not ideal, will convert below
            # convert to Series aligned with params
            bse = np.array(np.sqrt(np.diag(cov)))
            bse = type(params)(bse, index=params.index)
        else:
            bse = None

    # p-values: try to get; if not, compute from normal approximation using params/bse
    pvalues = None
    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = None

    if pvalues is None:
        if bse is None:
            pvalues = None
        else:
            z = params / bse
            # normal two-sided p-value using math.erf to avoid external deps
            def norm_cdf(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            pvals = []
            for zi in np.asarray(z):
                pvals.append(2.0 * (1.0 - norm_cdf(abs(zi))))
            pvalues = type(params)(pvals, index=params.index)

    # confidence intervals: try method
    try:
        ci_df = res.conf_int()
        # conf_int might return array-like; ensure we can index by param name
        # Convert to DataFrame-like: lower = ci_df[0], upper = ci_df[1]
        if hasattr(ci_df, 'loc') and params.index[0] in ci_df.index:
            ci_lower = ci_df.iloc[:, 0]
            ci_upper = ci_df.iloc[:, 1]
        else:
            # fallback: assume same order as params
            ci_lower = type(params)([ci_df[i, 0] for i in range(len(params))], index=params.index)
            ci_upper = type(params)([ci_df[i, 1] for i in range(len(params))], index=params.index)
    except Exception:
        # Build a 95% CI from params +/- 1.96*bse if bse available
        if bse is not None:
            ci_lower = params - 1.96 * bse
            ci_upper = params + 1.96 * bse
        else:
            ci_lower = None
            ci_upper = None

    # Helper to find parameter names robustly
    param_index = list(params.index.astype(str))

    def find_main_param(name):
        # find a parameter that contains the name but not the other moderator name
        matches = [p for p in param_index if (name in p) and (':' not in p)]
        if len(matches) == 1:
            return matches[0]
        # if multiple or none, prefer exact match
        if name in param_index:
            return name
        # otherwise return first that contains the name
        for p in param_index:
            if name in p:
                return p
        return None

    def find_interaction_param(a, b):
        # find parameter name that contains both a and b (order-agnostic)
        for p in param_index:
            if (a in p) and (b in p):
                return p
        for p in param_index:
            if (b in p) and (a in p):
                return p
        return None

    main_name = find_main_param('reader_view')
    inter_name = find_interaction_param('reader_view', 'dyslexia_bin')

    # Extract main coefficients
    def get_val(series, name):
        if name is None:
            return None
        try:
            return series[name]
        except Exception:
            # sometimes index types differ; try to find matching by substring
            for key in series.index:
                if str(name) == str(key):
                    return series[key]
            # nothing found
            return None

    coef_reader = get_val(params, main_name)
    p_reader = get_val(pvalues, main_name) if pvalues is not None else None
    ci_reader = (get_val(ci_lower, main_name), get_val(ci_upper, main_name)) if (ci_lower is not None) else (None, None)
    se_reader = get_val(bse, main_name) if bse is not None else None

    coef_inter = get_val(params, inter_name)
    p_inter = get_val(pvalues, inter_name) if pvalues is not None else None
    ci_inter = (get_val(ci_lower, inter_name), get_val(ci_upper, inter_name)) if (ci_lower is not None) else (None, None)
    se_inter = get_val(bse, inter_name) if bse is not None else None

    # Simple effects:
    # Non-dyslexic (dyslexia_bin = 0): effect = coef_reader
    effect_non = None
    effect_non_se = None
    effect_non_p = None
    effect_non_ci = (None, None)
    effect_non_pct = None
    if coef_reader is not None:
        effect_non = float(coef_reader)
        effect_non_se = float(se_reader) if se_reader is not None else None
        effect_non_p = float(p_reader) if p_reader is not None else None
        effect_non_ci = (float(ci_reader[0]) if ci_reader[0] is not None else None,
                         float(ci_reader[1]) if ci_reader[1] is not None else None)
        try:
            effect_non_pct = (math.exp(effect_non) - 1.0) * 100.0
        except Exception:
            effect_non_pct = None

    # Dyslexic (dyslexia_bin = 1): effect = coef_reader + coef_inter
    effect_dys = None
    effect_dys_se = None
    effect_dys_p = None
    effect_dys_ci = (None, None)
    effect_dys_pct = None
    if coef_reader is not None:
        if coef_inter is not None:
            effect_dys = float(coef_reader + coef_inter)
            # SE of sum: var = var(a) + var(b) + 2*cov(a,b)
            if cov is not None:
                # try to access elements of covariance matrix robustly
                try:
                    var_main = cov.loc[main_name, main_name]
                    var_inter = cov.loc[inter_name, inter_name]
                    cov_main_inter = cov.loc[main_name, inter_name]
                except Exception:
                    # fallback to positional indexing
                    try:
                        idx_main = param_index.index(main_name)
                        idx_inter = param_index.index(inter_name)
                        var_main = cov.iloc[idx_main, idx_main]
                        var_inter = cov.iloc[idx_inter, idx_inter]
                        cov_main_inter = cov.iloc[idx_main, idx_inter]
                    except Exception:
                        var_main = var_inter = cov_main_inter = None
                if (var_main is not None) and (var_inter is not None) and (cov_main_inter is not None):
                    var_sum = var_main + var_inter + 2.0 * cov_main_inter
                    effect_dys_se = float(math.sqrt(var_sum)) if var_sum >= 0 else None
                    # p-value using normal approx
                    if effect_dys_se is not None and effect_dys_se > 0:
                        z = effect_dys / effect_dys_se
                        # normal cdf
                        def norm_cdf(x):
                            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                        effect_dys_p = 2.0 * (1.0 - norm_cdf(abs(z)))
                        # 95% CI
                        lower = effect_dys - 1.96 * effect_dys_se
                        upper = effect_dys + 1.96 * effect_dys_se
                        effect_dys_ci = (lower, upper)
                else:
                    # can't compute SE from cov
                    effect_dys_se = None
                    effect_dys_p = None
                    effect_dys_ci = (None, None)
            else:
                # no cov available, try approximate via bse if independent (not ideal)
                if (se_reader is not None) and (se_inter is not None):
                    effect_dys_se = float(math.sqrt(se_reader**2 + se_inter**2))
                    def norm_cdf(x):
                        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                    z = effect_dys / effect_dys_se
                    effect_dys_p = 2.0 * (1.0 - norm_cdf(abs(z)))
                    effect_dys_ci = (effect_dys - 1.96 * effect_dys_se, effect_dys + 1.96 * effect_dys_se)
                else:
                    effect_dys_se = None
            try:
                effect_dys_pct = (math.exp(effect_dys) - 1.0) * 100.0
            except Exception:
                effect_dys_pct = None
        else:
            # no interaction present -> effect same as main effect
            effect_dys = effect_non
            effect_dys_se = effect_non_se
            effect_dys_p = effect_non_p
            effect_dys_ci = effect_non_ci
            effect_dys_pct = effect_non_pct

    # Prepare output dict
    output = {
        "coef_reader_view": float(coef_reader) if coef_reader is not None else None,
        "se_reader_view": float(se_reader) if se_reader is not None else None,
        "p_reader_view": float(p_reader) if p_reader is not None else None,
        "ci_reader_view": (float(ci_reader[0]) if ci_reader[0] is not None else None,
                           float(ci_reader[1]) if ci_reader[1] is not None else None),
        "coef_interaction": float(coef_inter) if coef_inter is not None else None,
        "se_interaction": float(se_inter) if se_inter is not None else None,
        "p_interaction": float(p_inter) if p_inter is not None else None,
        "ci_interaction": (float(ci_inter[0]) if ci_inter[0] is not None else None,
                           float(ci_inter[1]) if ci_inter[1] is not None else None),
        "effect_non_dyslexic": {
            "coef": effect_non,
            "se": effect_non_se,
            "p": effect_non_p,
            "ci_95": effect_non_ci,
            "pct_change_speed_plus1": effect_non_pct
        },
        "effect_dyslexic": {
            "coef": effect_dys,
            "se": effect_dys_se,
            "p": effect_dys_p,
            "ci_95": effect_dys_ci,
            "pct_change_speed_plus1": effect_dys_pct
        },
        # include raw param names used so user can inspect
        "param_names_used": {
            "main_name": main_name,
            "interaction_name": inter_name
        }
    }

    # Short description
    description_lines = [
        "Returned items:",
        "- coef_reader_view: estimated effect of activating Reader View on log(speed+1) for the reference group (dyslexia_bin=0).",
        "- coef_interaction: additional change in the Reader View effect for dyslexic readers (so Reader View effect for dyslexic = coef_reader_view + coef_interaction).",
        "- effect_* entries: simple (group-specific) effects converted to percent change on the speed+1 scale via (exp(coef)-1)*100.",
        "",
        "Interpretation guidance:",
        "- If coef_reader_view (or effect_dyslexic) is positive and statistically significant (small p), Reader View is associated with higher log-speed -> higher speed.",
        "- The interaction term tests whether the Reader View effect differs for dyslexic readers; a significant positive interaction indicates a larger Reader View benefit for dyslexic readers, a significant negative interaction indicates a smaller benefit (or a cost).",
        "- Percent changes are on speed+1 due to the log(speed+1) transformation; approximate percent change in reported speed is shown in pct_change_speed_plus1.",
        "",
        "Notes on robustness:",
        "- If the model object did not expose covariance (cov_params), the SE and p-value for the dyslexic simple effect may be unavailable (None).",
        "- p-values are taken from the model if available; otherwise they are approximated by a normal (z) approximation using the estimated SE."
    ]
    description = "\n".join(description_lines)

    return {"object": output, "description": description}