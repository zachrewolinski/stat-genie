def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    interpretable (multiplicative) effects for:
      - the main effect of reader_view (effect for non-dyslexic when dyslexia_bin=0)
      - the interaction reader_view:dyslexia_bin
      - the combined effect for dyslexic participants (reader_view + interaction)

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Plain-language summary of results and interpretation"
      }
    """
    import math
    import numpy as np

    # Try to import a normal distribution helper; fall back to math.erf if scipy is not available
    try:
        from scipy import stats
        norm_sf = lambda x: stats.norm.sf(x)
    except Exception:
        # survival function using erf: sf(x) = 0.5*(1 - erf(x/sqrt(2)))
        norm_sf = lambda x: 0.5 * (1.0 - math.erf(x / math.sqrt(2.0)))

    res = model_output

    # Extract basic objects (params, bse, pvalues, conf_int, cov_params)
    try:
        params = res.params
    except Exception as e:
        raise ValueError("Could not extract params from model_output: " + str(e))

    # Convert index to list of param names
    param_names = list(params.index)

    # Helper to find parameter name by substrings (robust to slight naming differences)
    def find_param(*substrings):
        matches = []
        for n in param_names:
            if all(sub in n for sub in substrings):
                matches.append(n)
        return matches

    # Find main effect name for reader_view (prefer name without ':' to avoid matching interactions)
    reader_candidates = [n for n in param_names if 'reader_view' in n and ':' not in n and 'reader_view[' not in n]
    if not reader_candidates:
        # fallback: any parameter containing 'reader_view'
        reader_candidates = [n for n in param_names if 'reader_view' in n]
    if not reader_candidates:
        raise ValueError(f"Could not find a parameter for 'reader_view' in model params: {param_names}")
    reader_name = reader_candidates[0]

    # Find interaction between reader_view and dyslexia_bin
    # Common statmodels naming: 'reader_view:dyslexia_bin'
    interaction_candidates = find_param('reader_view', 'dyslexia')
    interaction_name = interaction_candidates[0] if interaction_candidates else None

    # Safely extract other result summaries
    pvalues = getattr(res, 'pvalues', None)
    bse = getattr(res, 'bse', None)
    try:
        conf = res.conf_int()  # DataFrame-like, rows indexed by param names, two columns
    except Exception:
        conf = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Helper to safely read a value from pandas Series-like objects
    def safe_get(series_like, name):
        if series_like is None:
            return None
        try:
            return float(series_like[name])
        except Exception:
            # maybe no exact match; try contains
            for n in series_like.index:
                if name == n:
                    return float(series_like[n])
            # not found
            return None

    # Main reader_view effect (this is the effect on log_speed for reader_view ON vs OFF when dyslexia_bin=0)
    beta_reader = safe_get(params, reader_name)
    se_reader = safe_get(bse, reader_name)
    p_reader = safe_get(pvalues, reader_name)
    ci_reader = None
    if conf is not None and reader_name in conf.index:
        ci_reader = [float(conf.loc[reader_name, 0]), float(conf.loc[reader_name, 1])]

    # Interaction effect
    if interaction_name:
        beta_int = safe_get(params, interaction_name)
        se_int = safe_get(bse, interaction_name)
        p_int = safe_get(pvalues, interaction_name)
        ci_int = None
        if conf is not None and interaction_name in conf.index:
            ci_int = [float(conf.loc[interaction_name, 0]), float(conf.loc[interaction_name, 1])]
    else:
        beta_int = 0.0
        se_int = None
        p_int = None
        ci_int = None

    # Combined effect for dyslexic participants (reader_view effect when dyslexia_bin=1)
    beta_dys = None
    se_dys = None
    p_dys = None
    ci_dys = None
    if beta_reader is not None:
        # compute combined coefficient (beta_reader + beta_interaction)
        beta_dys = beta_reader + (beta_int if beta_int is not None else 0.0)

        # compute SE for the sum using covariance matrix if available
        if cov is not None and interaction_name is not None:
            try:
                var_reader = float(cov.loc[reader_name, reader_name])
                var_int = float(cov.loc[interaction_name, interaction_name])
                cov_ri = float(cov.loc[reader_name, interaction_name])
                var_sum = var_reader + var_int + 2.0 * cov_ri
                se_dys = math.sqrt(var_sum) if var_sum >= 0 else None
            except Exception:
                se_dys = None
        else:
            # fallback: approximate by sqrt(se_reader^2 + se_int^2) ignoring covariance if both SEs available
            if se_reader is not None and se_int is not None:
                se_dys = math.sqrt(se_reader ** 2 + se_int ** 2)
            else:
                se_dys = None

        # p-value (Wald z using normal approximation) if SE available
        if se_dys is not None and se_dys > 0:
            z = beta_dys / se_dys
            p_dys = float(2.0 * norm_sf(abs(z)))
            # 95% CI for beta_dys using normal approx
            zcrit = 1.96
            ci_dys = [beta_dys - zcrit * se_dys, beta_dys + zcrit * se_dys]

    # Convert log-scale effects to multiplicative effects on raw speed (exp(coef))
    mult_reader = math.exp(beta_reader) if beta_reader is not None else None
    mult_reader_ci = None
    if ci_reader is not None:
        mult_reader_ci = [math.exp(ci_reader[0]), math.exp(ci_reader[1])]

    mult_dys = math.exp(beta_dys) if beta_dys is not None else None
    mult_dys_ci = None
    if ci_dys is not None:
        mult_dys_ci = [math.exp(ci_dys[0]), math.exp(ci_dys[1])]

    # Prepare a concise human-readable interpretation
    # Interpretations: percent change = (exp(beta)-1)*100
    def pct_change_from_mult(m):
        return (m - 1.0) * 100.0 if m is not None else None

    reader_pct = pct_change_from_mult(mult_reader)
    reader_pct_ci = [pct_change_from_mult(x) for x in mult_reader_ci] if mult_reader_ci else None
    dys_pct = pct_change_from_mult(mult_dys)
    dys_pct_ci = [pct_change_from_mult(x) for x in mult_dys_ci] if mult_dys_ci else None

    # Build the object to return
    object_result = {
        "reader_view_param_name": reader_name,
        "reader_view": {
            "beta_log": beta_reader,
            "se": se_reader,
            "p_value": p_reader,
            "ci_95_log": ci_reader,
            "multiplicative_effect_on_speed": mult_reader,            # exp(beta)
            "percent_change_on_speed": reader_pct,                    # (exp(beta)-1)*100
            "multiplicative_ci_95": mult_reader_ci,
            "percent_change_ci_95": reader_pct_ci
        },
        "interaction_param_name": interaction_name,
        "interaction": {
            "beta_log": beta_int,
            "se": se_int,
            "p_value": p_int,
            "ci_95_log": ci_int
        },
        "dyslexic_effect": {
            "beta_log": beta_dys,
            "se": se_dys,
            "p_value": p_dys,
            "ci_95_log": ci_dys,
            "multiplicative_effect_on_speed": mult_dys,
            "percent_change_on_speed": dys_pct,
            "multiplicative_ci_95": mult_dys_ci,
            "percent_change_ci_95": dys_pct_ci
        },
        # Optional: include raw params, pvalues for inspection
        "raw": {
            "params": params.to_dict() if hasattr(params, "to_dict") else dict(params),
            "pvalues": (pvalues.to_dict() if (pvalues is not None and hasattr(pvalues, "to_dict")) else (dict(pvalues) if pvalues is not None else None)),
        }
    }

    # Construct a brief description
    # We will state whether there is evidence that reader_view changes reading speed for dyslexic individuals
    desc_lines = []

    if beta_dys is not None:
        desc_lines.append(
            "Estimated effect of turning Reader View ON for dyslexic participants (on log reading speed): "
            f"{beta_dys:.4g}"
            + (f" (SE = {se_dys:.4g})" if se_dys is not None else "")
            + (f", 95% CI = [{ci_dys[0]:.4g}, {ci_dys[1]:.4g}]" if ci_dys is not None else "")
        )
        if dys_pct is not None:
            desc_lines.append(
                f"On the original speed scale this corresponds to a multiplicative effect of {mult_dys:.3g}x "
                f"({dys_pct:.1f}% change; 95% CI ~ [{dys_pct_ci[0]:.1f}%, {dys_pct_ci[1]:.1f}%])"
                if dys_pct_ci is not None else
                f"On the original speed scale this corresponds to a multiplicative effect of {mult_dys:.3g}x ({dys_pct:.1f}% change)"
            )
        if p_dys is not None:
            sig = "statistically significant" if p_dys < 0.05 else "not statistically significant"
            desc_lines.append(f"Statistical test (Wald normal approximation) p = {p_dys:.3g} → {sig}.")
        else:
            desc_lines.append("P-value for combined effect could not be computed.")
    else:
        desc_lines.append("Could not compute the combined effect for dyslexic participants due to missing coefficients.")

    # Also report the interaction term significance directly (whether the reader_view effect differs by dyslexia)
    if interaction_name is not None:
        if p_int is not None:
            sig = "statistically significant" if p_int < 0.05 else "not statistically significant"
            desc_lines.append(
                f"The interaction term ({interaction_name}) has beta = {beta_int:.4g}, p = {p_int:.3g} → {sig}. "
                "A statistically significant interaction means the effect of Reader View differs between dyslexic and non-dyslexic participants."
            )
        else:
            desc_lines.append(f"Interaction term ({interaction_name}) found but p-value not available.")
    else:
        desc_lines.append("No explicit interaction parameter found in the model output.")

    description = " ".join(desc_lines)

    return {"object": object_result, "description": description}