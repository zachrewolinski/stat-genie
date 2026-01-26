def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, and 95% CI (and their exponentiated IRRs)
    for the predictors of interest (age, sex_m, help_y) from the supplied model_output dict.

    Returns a dict with keys:
      - "object": dict containing model_type, overdispersion, and per-variable stats:
          { var_name: {
                'coef': log-scale coefficient,
                'se': standard error,
                'pvalue': p-value,
                'IRR': exp(coef),
                'IRR_CI_low': exp(lower CI),
                'IRR_CI_high': exp(upper CI),
                'pct_change': (IRR-1)*100  # percent change in rate
            }, ... }
      - "description": brief human-readable interpretation of the variables' effects.
    """
    import numpy as np

    # Accept either the full dict returned by the modeling function or a raw results object
    if isinstance(model_output, dict):
        res = model_output.get('result') or model_output.get('raw_result')
        model_type = model_output.get('model_type')
        overdispersion = model_output.get('overdispersion')
    else:
        # assume it's a statsmodels results object
        res = model_output
        model_type = None
        overdispersion = None

    if res is None:
        raise ValueError("No fitted results object found in model_output.")

    # Variables of interest
    vars_of_interest = ['age', 'sex_m', 'help_y']

    # Try to obtain parameter table and conf_int using the results object.
    # Many statsmodels ResultsWrapper objects expose: params, bse, pvalues, conf_int()
    params = getattr(res, 'params', None)
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    try:
        conf = res.conf_int()  # default 95% CI
    except Exception:
        # fallback: attempt to compute from params +/- 1.96*bse if possible
        conf = None

    var_stats = {}
    for v in vars_of_interest:
        if params is None or v not in params.index:
            var_stats[v] = {'present': False, 'note': f"Variable '{v}' not found in model parameters."}
            continue

        coef = float(params.loc[v])
        se = float(bse.loc[v]) if (bse is not None and v in bse.index) else None
        pval = float(pvalues.loc[v]) if (pvalues is not None and v in pvalues.index) else None

        # Confidence interval (log scale)
        if conf is not None and v in conf.index:
            ci_low_log = float(conf.loc[v, 0])
            ci_high_log = float(conf.loc[v, 1])
        elif se is not None:
            # approximate using normal approx
            ci_low_log = coef - 1.96 * se
            ci_high_log = coef + 1.96 * se
        else:
            ci_low_log = ci_high_log = None

        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low_log)) if ci_low_log is not None else None
        irr_ci_high = float(np.exp(ci_high_log)) if ci_high_log is not None else None
        pct_change = (irr - 1.0) * 100.0

        var_stats[v] = {
            'present': True,
            'coef': coef,
            'se': se,
            'pvalue': pval,
            'IRR': irr,
            'IRR_CI_low': irr_ci_low,
            'IRR_CI_high': irr_ci_high,
            'pct_change': pct_change,
        }

    # Build a concise human-readable description
    desc_lines = []
    if model_type:
        desc_lines.append(f"Model type: {model_type}.")
    if overdispersion is not None:
        desc_lines.append(f"Overdispersion (deviance / df_resid): {overdispersion:.3f}.")

    for v in vars_of_interest:
        info = var_stats[v]
        if not info.get('present'):
            desc_lines.append(info.get('note'))
            continue
        irr = info['IRR']
        low = info['IRR_CI_low']
        high = info['IRR_CI_high']
        pval = info['pvalue']
        pct = info['pct_change']

        if v == 'age':
            var_label = "Age (per year)"
            interp = f"Each additional year is associated with an IRR = {irr:.3f}"
            if low is not None and high is not None:
                interp += f" (95% CI {low:.3f}–{high:.3f})"
            interp += f", i.e. approximately {pct:.2f}% change in nut-opening rate per year"
        elif v == 'sex_m':
            var_label = "Sex (male vs female)"
            interp = f"Males vs females: IRR = {irr:.3f}"
            if low is not None and high is not None:
                interp += f" (95% CI {low:.3f}–{high:.3f})"
            interp += f", i.e. males have about {pct:.2f}% difference in rate compared to females"
        else:  # help_y
            var_label = "Received help (yes vs no)"
            interp = f"Receiving help: IRR = {irr:.3f}"
            if low is not None and high is not None:
                interp += f" (95% CI {low:.3f}–{high:.3f})"
            interp += f", i.e. sessions with help have about {pct:.2f}% difference in nut-opening rate"

        if pval is not None:
            interp += f"; p = {pval:.3f}."
        else:
            interp += "."

        desc_lines.append(f"{var_label}: {interp}")

    description = " ".join(desc_lines)

    return {
        "object": {
            "model_type": model_type,
            "overdispersion": overdispersion,
            "variables": var_stats
        },
        "description": description
    }