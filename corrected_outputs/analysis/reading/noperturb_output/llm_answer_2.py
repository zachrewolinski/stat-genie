def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and confidence intervals for:
      - reader_view (main effect: effect of Reader View for non-dyslexic readers)
      - reader_view + reader_view_x_dyslexia (effect of Reader View for readers with dyslexia)
      - reader_view_x_dyslexia (interaction term)

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Concise interpretation in context"
      }
    """
    import numpy as np
    import math
    from scipy import stats

    # Defensive checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object (missing .params)")

    params = model_output.params
    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        # fallback to results normalized_cov_params if cov_params not available
        try:
            cov = model_output.normalized_cov_params
        except Exception:
            cov = None

    # Helper to safely get param value
    def get_param(name):
        return float(params[name]) if name in params.index else None

    beta_rv = get_param('reader_view')
    beta_int = get_param('reader_view_x_dyslexia')
    # Get direct interaction coefficient if available name slightly different?
    # (We assume the model code used exactly 'reader_view_x_dyslexia'.)
    # Standard errors and p-values from params and bse if available
    bse = None
    pvals = None
    conf_int = None
    try:
        bse = model_output.bse
        pvals = model_output.pvalues
        conf_int = model_output.conf_int()
    except Exception:
        bse = None

    # Prepare results structure
    out = {}
    nobs = getattr(model_output, "nobs", None)
    out['nobs'] = int(nobs) if nobs is not None else None

    # Function to compute linear combination stats: coef, se, z/t, p, ci
    def lincomb_stats(coef_names, human_name):
        # coef_names: list of parameter names to sum
        # Return dict with coef (log-units), se, tstat, pval, ci (95%), percent changes
        # Check existence
        coefs = []
        missing = []
        for nm in coef_names:
            if nm in params.index:
                coefs.append(float(params[nm]))
            else:
                missing.append(nm)
        if missing:
            return {'present': False, 'missing_params': missing}

        coef_sum = sum(coefs)

        # compute variance of linear combination using cov matrix if available
        se = None
        ci = None
        pval = None
        tstat = None
        if cov is not None:
            # Build vector
            vec = np.zeros(len(params))
            for nm in coef_names:
                idx = list(params.index).index(nm)
                vec[idx] += 1.0
            var = float(vec @ cov.values @ vec) if hasattr(cov, "values") else float(vec @ cov @ vec)
            se = math.sqrt(var) if var >= 0 else float('nan')
            if se == 0 or math.isnan(se):
                tstat = None
                pval = None
            else:
                tstat = coef_sum / se
                # Use normal approx for two-sided p-value if df not available; prefer t with large df
                # Try to get df_resid
                df_resid = getattr(model_output, "df_resid", None)
                if df_resid is not None and df_resid > 0:
                    pval = 2 * stats.t.sf(abs(tstat), df_resid)
                else:
                    pval = 2 * stats.norm.sf(abs(tstat))
            # 95% CI
            if se is not None:
                if df_resid is not None and df_resid > 0:
                    crit = stats.t.ppf(0.975, df_resid)
                else:
                    crit = stats.norm.ppf(0.975)
                ci = (coef_sum - crit * se, coef_sum + crit * se)
        else:
            # Try to use model_output.t_test if cov not usable
            try:
                if len(coef_names) == 1:
                    # simple t_test like "param = 0"
                    t_res = model_output.t_test(f"{coef_names[0]} = 0")
                else:
                    # build expression like "reader_view + reader_view_x_dyslexia = 0"
                    expr = " + ".join(coef_names) + " = 0"
                    t_res = model_output.t_test(expr)
                coef_sum = float(t_res.effect.squeeze())
                se = float(t_res.sd.squeeze()) if hasattr(t_res, "sd") else None
                pval = float(t_res.pvalue)
                if hasattr(t_res, "conf_int"):
                    ci_arr = t_res.conf_int()
                    ci = (float(ci_arr[0, 0]), float(ci_arr[0, 1]))
                if se and se != 0:
                    tstat = coef_sum / se
            except Exception:
                pass

        # Percent interpretations
        approx_pct = 100.0 * coef_sum if coef_sum is not None else None
        exact_pct = (100.0 * (math.exp(coef_sum) - 1.0)) if coef_sum is not None else None
        ci_pct = None
        if ci is not None:
            ci_pct = (100.0 * (math.exp(ci[0]) - 1.0), 100.0 * (math.exp(ci[1]) - 1.0))

        return {
            'present': True,
            'name': human_name,
            'coef_log_units': coef_sum,
            'se': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_log_units_95%': ci,
            'approx_pct_change': approx_pct,   # 100 * beta, approximate
            'exact_pct_change': exact_pct,     # 100*(exp(beta)-1)
            'ci_pct_change_95%': ci_pct
        }

    # Stats for non-dyslexic effect (reader_view alone)
    if 'reader_view' in params.index:
        out['reader_view_non_dyslexic'] = lincomb_stats(['reader_view'], 'Reader View effect (non-dyslexic)')
    else:
        out['reader_view_non_dyslexic'] = {'present': False, 'missing_params': ['reader_view']}

    # Stats for interaction term alone
    if 'reader_view_x_dyslexia' in params.index:
        out['interaction_readerview_dyslexia'] = {
            'present': True,
            'coef_log_units': float(params['reader_view_x_dyslexia']),
            'se': float(bse['reader_view_x_dyslexia']) if (bse is not None and 'reader_view_x_dyslexia' in bse.index) else None,
            'p_value': float(pvals['reader_view_x_dyslexia']) if (pvals is not None and 'reader_view_x_dyslexia' in pvals.index) else None,
            'ci_log_units_95%': tuple(conf_int.loc['reader_view_x_dyslexia'].values) if (conf_int is not None and 'reader_view_x_dyslexia' in conf_int.index) else None,
            'approx_pct_change': 100.0 * float(params['reader_view_x_dyslexia']),
            'exact_pct_change': 100.0 * (math.exp(float(params['reader_view_x_dyslexia'])) - 1.0)
        }
    else:
        out['interaction_readerview_dyslexia'] = {'present': False, 'missing_params': ['reader_view_x_dyslexia']}

    # Stats for dyslexic effect: reader_view + interaction
    if ('reader_view' in params.index) and ('reader_view_x_dyslexia' in params.index):
        out['reader_view_dyslexic'] = lincomb_stats(['reader_view', 'reader_view_x_dyslexia'],
                                                    'Reader View effect (dyslexic)')
    else:
        missing = []
        if 'reader_view' not in params.index:
            missing.append('reader_view')
        if 'reader_view_x_dyslexia' not in params.index:
            missing.append('reader_view_x_dyslexia')
        out['reader_view_dyslexic'] = {'present': False, 'missing_params': missing}

    # Short human-readable interpretation
    # Build interpretation using available pieces
    def fmt(x, nd=3):
        return (f"{x:.{nd}f}" if (x is not None and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else str(x))

    desc_lines = []
    desc_lines.append(f"Model has {out.get('nobs', 'N/A')} observations used for estimation.")
    # Non-dyslexic
    nd = out['reader_view_non_dyslexic']
    if nd.get('present'):
        desc_lines.append(
            "For readers without dyslexia, Reader View effect (log units) = "
            f"{fmt(nd['coef_log_units'])}; approx % change = {fmt(nd['approx_pct_change'])}%; "
            f"exact % change = {fmt(nd['exact_pct_change'])}% ; p = {fmt(nd['p_value'])}."
        )
    else:
        desc_lines.append("Reader View main effect not available in model output.")

    # Dyslexic
    dd = out['reader_view_dyslexic']
    if dd.get('present'):
        desc_lines.append(
            "For readers with dyslexia, Reader View effect (log units) = "
            f"{fmt(dd['coef_log_units'])}; approx % change = {fmt(dd['approx_pct_change'])}%; "
            f"exact % change = {fmt(dd['exact_pct_change'])}% ; p = {fmt(dd['p_value'])}."
        )
    else:
        desc_lines.append("Reader View effect for dyslexic readers could not be computed (missing terms).")

    # Interaction interpretation
    inter = out['interaction_readerview_dyslexia']
    if inter.get('present'):
        desc_lines.append(
            "Interaction (reader_view x dyslexia) coefficient (log units) = "
            f"{fmt(inter['coef_log_units'])}; approx % change = {fmt(inter['approx_pct_change'])}% ; "
            f"p = {fmt(inter['p_value'])}."
        )
        # Quick interpretation sentence
        # If coefficient for dyslexic effect and interaction both present and p-values exist, assess significance
        try:
            p_inter = inter.get('p_value')
            coef_inter = inter.get('coef_log_units')
            if p_inter is not None and p_inter < 0.05:
                sign_text = "statistically significant"
            else:
                sign_text = "not statistically significant"
            desc_lines.append(f"The interaction is {sign_text} (p = {fmt(p_inter)}).")
        except Exception:
            pass
    else:
        desc_lines.append("Interaction term not present; cannot assess differential effect by dyslexia status.")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}