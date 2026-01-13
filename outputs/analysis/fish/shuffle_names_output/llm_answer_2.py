def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels OLS RegressionResultsWrapper fit to:
      LogCatchRate ~ log_persons + livebait + child + camper + livebait:log_persons

    Returns a dictionary with:
      - "object": dict of extracted numeric summaries (coefficients, SEs, p-values,
                  95% CIs, combined/interaction effects, and implied percent changes)
      - "description": human-readable interpretation of those numbers in context

    The function is written defensively to handle slight variations in parameter names
    (e.g., interaction name could be 'livebait:log_persons' or 'log_persons:livebait',
    and livebait main term could appear with categorical naming).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic parameter table
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int(alpha=0.05)
    conf.columns = ['ci_lower', 'ci_upper']

    # helper to find parameter name by containing substrings
    def find_param(name_parts):
        # name_parts: list of strings that must all appear in the param name (case sensitive)
        for name in params.index:
            if all(part in name for part in name_parts):
                return name
        return None

    # identify parameter names robustly
    name_logp = find_param(['log_persons']) or find_param(['log(persons)', 'log_persons'])
    name_live = find_param(['livebait'])
    name_inter = None
    # try specific interaction patterns
    cand1 = find_param(['livebait', 'log_persons'])
    cand2 = find_param(['log_persons', 'livebait'])
    name_inter = cand1 or cand2

    # prepare output container
    out = {'params': {}, 'combined_effects': {}, 'percent_changes': {}}

    # fill param info if present
    for logical_name, pname in [('log_persons', name_logp), ('livebait', name_live), ('interaction', name_inter)]:
        if pname is not None and pname in params.index:
            out['params'][logical_name] = {
                'param_name': pname,
                'coef': float(params[pname]),
                'se': float(bse[pname]),
                'pvalue': float(pvals[pname]),
                'ci_lower': float(conf.loc[pname, 'ci_lower']),
                'ci_upper': float(conf.loc[pname, 'ci_upper'])
            }
        else:
            out['params'][logical_name] = None

    # covariance matrix for computing joint variances
    cov = res.cov_params()

    # Interpretation helpers
    ln2 = np.log(2.0)

    # 1) Effect (elasticity) of log_persons on LogCatchRate when livebait=0
    if out['params']['log_persons'] is not None:
        beta_lp = out['params']['log_persons']['coef']
        se_lp = out['params']['log_persons']['se']
        ci_lp = (out['params']['log_persons']['ci_lower'], out['params']['log_persons']['ci_upper'])
        out['combined_effects']['logpersons_livebait0'] = {
            'coef_on_log_catchrate': beta_lp,
            'se': se_lp,
            '95ci': ci_lp,
            'interpretation': (
                'Elasticity: percent change in catch rate for a 1% change in group size '
                '(when livebait=0) approximately equals this coefficient × 1%'
            )
        }
        # percent change for doubling persons (livebait=0)
        pct_double = (np.exp(beta_lp * ln2) - 1.0) * 100.0
        # approximate 95% CI for beta then transform
        z = 1.96
        lower_beta = beta_lp - z * se_lp
        upper_beta = beta_lp + z * se_lp
        pct_double_ci = (
            (np.exp(lower_beta * ln2) - 1.0) * 100.0,
            (np.exp(upper_beta * ln2) - 1.0) * 100.0
        )
        out['percent_changes']['doubling_persons_livebait0'] = {
            'percent_change': float(pct_double),
            '95ci_percent_change': (float(pct_double_ci[0]), float(pct_double_ci[1])),
            'note': 'Percent change in catch rate when group size doubles (livebait=0).'
        }
    else:
        out['combined_effects']['logpersons_livebait0'] = None

    # 2) Effect of log_persons when livebait=1 (combined coefficient = beta_lp + beta_inter)
    if out['params']['log_persons'] is not None and out['params']['interaction'] is not None:
        beta_lp = out['params']['log_persons']['coef']
        beta_int = out['params']['interaction']['coef']
        # find actual parameter names for covariance retrieval
        pname_lp = out['params']['log_persons']['param_name']
        pname_int = out['params']['interaction']['param_name']
        coef_comb = beta_lp + beta_int
        # variance: Var(beta_lp) + Var(beta_int) + 2*Cov(beta_lp, beta_int)
        var_comb = cov.loc[pname_lp, pname_lp] + cov.loc[pname_int, pname_int] + 2.0 * cov.loc[pname_lp, pname_int]
        se_comb = float(np.sqrt(var_comb))
        ci_comb = (coef_comb - 1.96 * se_comb, coef_comb + 1.96 * se_comb)
        out['combined_effects']['logpersons_livebait1'] = {
            'coef_on_log_catchrate': float(coef_comb),
            'se': se_comb,
            '95ci': (float(ci_comb[0]), float(ci_comb[1])),
            'interpretation': 'Elasticity of catch rate w.r.t. group size when livebait=1 (log scale).'
        }
        # percent change for doubling persons when livebait=1
        pct_double = (np.exp(coef_comb * ln2) - 1.0) * 100.0
        pct_double_ci = (
            (np.exp(ci_comb[0] * ln2) - 1.0) * 100.0,
            (np.exp(ci_comb[1] * ln2) - 1.0) * 100.0
        )
        out['percent_changes']['doubling_persons_livebait1'] = {
            'percent_change': float(pct_double),
            '95ci_percent_change': (float(pct_double_ci[0]), float(pct_double_ci[1])),
            'note': 'Percent change in catch rate when group size doubles (livebait=1).'
        }
    else:
        out['combined_effects']['logpersons_livebait1'] = None

    # 3) Effect of switching from livebait=0 to livebait=1 at a representative log_persons (median if available)
    # Need a value for log_persons; try to get the data used to fit the model
    median_logp = None
    try:
        df = None
        # attempt to get the original dataframe used in the model
        if hasattr(res.model, 'data'):
            # statsmodels often stores 'frame' or 'orig_endog' structures
            if hasattr(res.model.data, 'frame') and res.model.data.frame is not None:
                df = res.model.data.frame
            elif hasattr(res.model.data, 'orig_endog'):
                # fallback
                df = None
        if df is not None and name_logp in df.columns:
            median_logp = float(df[name_logp].median())
    except Exception:
        median_logp = None

    # if median not available, use 0 (interpretable but not ideal). We'll include a flag.
    if median_logp is None:
        # use 0 as baseline (interpretable as effect at log_persons=0 => persons = 1)
        median_logp = 0.0
        median_note = 'median log_persons not available from model data; used 0 (i.e., 1 person) as reference'
    else:
        median_note = f'median log_persons = {median_logp:.4g} computed from model data'

    if out['params']['livebait'] is not None and out['params']['interaction'] is not None:
        beta_live = out['params']['livebait']['coef']
        beta_int = out['params']['interaction']['coef']
        pname_live = out['params']['livebait']['param_name']
        pname_int = out['params']['interaction']['param_name']
        # marginal effect on log-catchrate of switching from 0->1 = beta_live + beta_int * log_persons_value
        delta = beta_live + beta_int * median_logp
        # variance:
        var_delta = cov.loc[pname_live, pname_live] + (median_logp**2) * cov.loc[pname_int, pname_int] + 2.0 * median_logp * cov.loc[pname_live, pname_int]
        se_delta = float(np.sqrt(var_delta))
        ci_delta = (delta - 1.96 * se_delta, delta + 1.96 * se_delta)
        pct_change = (np.exp(delta) - 1.0) * 100.0
        pct_ci = ((np.exp(ci_delta[0]) - 1.0) * 100.0, (np.exp(ci_delta[1]) - 1.0) * 100.0)
        out['combined_effects']['livebait_switch_at_logpersons'] = {
            'log_persons_value_used': float(median_logp),
            'value_note': median_note,
            'delta_log_catchrate': float(delta),
            'se': se_delta,
            '95ci_delta': (float(ci_delta[0]), float(ci_delta[1])),
            'percent_change_in_catchrate': float(pct_change),
            '95ci_percent_change': (float(pct_ci[0]), float(pct_ci[1])),
            'interpretation': (
                'Estimated multiplicative change in catch rate when switching to livebait '
                f'at log_persons={median_logp:.4g}'
            )
        }
    else:
        out['combined_effects']['livebait_switch_at_logpersons'] = None

    # Add a concise parameter table (for main variables)
    param_table = {}
    for k in ['log_persons', 'livebait', 'interaction']:
        info = out['params'][k]
        param_table[k] = info
    out['param_table'] = param_table

    # Build a short textual description summarizing findings and how to interpret them
    description_lines = []
    description_lines.append(
        "This output summarizes the fitted OLS on ln(CatchRate). Coefficients are on the log scale."
    )
    description_lines.append(
        ("- Coefficient on log_persons is an elasticity: a 1% increase in group size changes catch rate "
         "by approximately (coef)% (multiplicative). Doubling group size multiplies catch rate by exp(coef * ln2).")
    )
    description_lines.append(
        "- The interaction allows the elasticity of group size to differ when livebait is used."
    )
    description_lines.append(
        "- The 'livebait' main effect (plus interaction*log_persons) gives the effect of using livebait "
        "on the log of catch rate at a given group size; transform by exp(...) - 1 to get percent change."
    )
    description_lines.append("Numeric results are provided in the 'object' field under 'params', 'combined_effects', and 'percent_changes'.")

    final = {
        "object": out,
        "description": " ".join(description_lines)
    }

    return final