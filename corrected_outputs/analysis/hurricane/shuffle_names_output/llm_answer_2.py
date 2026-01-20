def extract_final_answer(model_output):
    """
    Inspect fitted model objects returned by the modeling function and extract
    statistics that test the hypothesis:
      "Higher femininity of hurricane names -> greater human cost (deaths)."
    
    The function looks for the primary continuous predictor 'NameFemScore_c'
    and the alternate binary 'FemaleName' in any returned models (e.g., 'nb_model',
    'poisson_robust', 'ols_log_deaths_robust'). For each model where the variable
    is present the function extracts:
      - coefficient (on the model scale),
      - p-value (robust where available),
      - 95% confidence interval,
      - for count models (NB/Poisson): exponentiated coefficient and CI (multiplicative effect),
      - for OLS on log(deaths+1): approximate percent change = 100*(exp(coef)-1).
    
    It then chooses a primary verdict using the preferred count model (Negative Binomial
    if present) and returns a summary dictionary and a short textual interpretation.
    """
    import numpy as np
    import pandas as pd

    out = {}
    summary = {}
    # Variables of interest (primary then alternate)
    vars_of_interest = ['NameFemScore_c', 'FemaleName']

    # Helper to try to obtain inference results with robust cov if possible
    def get_infer_result(res):
        try:
            # many statsmodels result objects implement get_robustcov_results
            robust = res.get_robustcov_results(cov_type='HC3')
            return robust
        except Exception:
            return res

    # Collect stats per model and variable
    for mname, res in model_output.items():
        try:
            infer_res = get_infer_result(res)
        except Exception:
            infer_res = res
        # Some results store params as a pandas Series with index of variable names
        try:
            params_index = list(infer_res.params.index)
        except Exception:
            # fallback: try res.params
            try:
                params_index = list(res.params.index)
                infer_res = res
            except Exception:
                params_index = []

        model_stats = {}
        # Try to detect if this is a count model (has family with name including 'Negative' or 'Poisson')
        is_count_model = False
        try:
            famname = res.model.family.__class__.__name__.lower()
            if ('negative' in famname) or ('poisson' in famname):
                is_count_model = True
        except Exception:
            # if model key mentions poisson/nb, use that as hint
            key_low = mname.lower()
            if ('nb' in key_low) or ('poisson' in key_low):
                is_count_model = True

        for var in vars_of_interest:
            if var in params_index:
                try:
                    coef = float(infer_res.params[var])
                except Exception:
                    coef = float(res.params[var])
                # p-value
                try:
                    pval = float(infer_res.pvalues[var])
                except Exception:
                    pval = float(res.pvalues[var]) if hasattr(res, 'pvalues') else np.nan
                # conf int
                try:
                    ci = infer_res.conf_int().loc[var].values.astype(float)
                except Exception:
                    try:
                        ci = res.conf_int().loc[var].values.astype(float)
                    except Exception:
                        ci = np.array([np.nan, np.nan])
                entry = {
                    'coef': coef,
                    'p_value': pval,
                    'ci_95': [ci[0], ci[1]],
                    'is_count_model': bool(is_count_model)
                }
                if is_count_model:
                    # multiplicative effect on expected counts
                    entry['exp_coef'] = float(np.exp(coef))
                    entry['exp_ci_95'] = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
                    entry['interpretation'] = (
                        "Multiplicative effect on expected death counts: exp(coef). "
                        "E.g., exp(coef)=1.10 means ~10% higher expected deaths per unit increase."
                    )
                else:
                    # OLS on log(deaths+1): approximate percent change
                    entry['pct_change_approx'] = float((np.exp(coef) - 1.0) * 100.0)
                    entry['interpretation'] = (
                        "Approx. percent change in (deaths+1): 100*(exp(coef)-1). "
                        "coef in log space; positive coef => higher deaths."
                    )
                # significance flag at 0.05
                entry['significant_0.05'] = bool((not np.isnan(pval)) and (pval < 0.05))
                model_stats[var] = entry
        if model_stats:
            summary[mname] = model_stats

    out['object'] = summary

    # Form a concise description + final verdict using a priority for the primary test:
    # prefer NB/Poisson model when available, else use OLS on log deaths.
    verdict = None
    desc_lines = []
    # Add per-model summaries
    for mname, vars_stats in summary.items():
        desc_lines.append(f"Model: {mname}")
        for var, st in vars_stats.items():
            line = (
                f"  Variable: {var} | coef={st['coef']:.4f} | p={st['p_value']:.3g} | "
                f"95%CI=[{st['ci_95'][0]:.4f}, {st['ci_95'][1]:.4f}]"
            )
            if st['is_count_model']:
                line += f" | exp(coef)={st['exp_coef']:.4f} | exp(95%CI)=[{st['exp_ci_95'][0]:.4f}, {st['exp_ci_95'][1]:.4f}]"
            else:
                line += f" | approx pct change={(st['pct_change_approx']):.2f}%"
            line += " | significant" if st['significant_0.05'] else " | not significant"
            desc_lines.append(line)
    # Decide primary verdict
    primary_order = ['nb_model', 'poisson_robust', 'poisson', 'ols_log_deaths_robust', 'ols']
    primary_choice = None
    for key in primary_order:
        if key in summary:
            primary_choice = (key, summary[key])
            break
    if primary_choice is None and len(summary) > 0:
        # fallback to any model present (first)
        primary_choice = next(iter(summary.items()))
    if primary_choice is None:
        final_text = "No models in model_output contained the variables of interest; cannot draw a conclusion."
    else:
        mname, vars_stats = primary_choice
        if 'NameFemScore_c' in vars_stats:
            st = vars_stats['NameFemScore_c']
            sign = "positive" if st['coef'] > 0 else "negative"
            sig = st['significant_0.05']
            if st['is_count_model']:
                effect_desc = f"exp(coef)={st['exp_coef']:.3f} (95%CI [{st['exp_ci_95'][0]:.3f}, {st['exp_ci_95'][1]:.3f}])"
            else:
                effect_desc = f"coef={st['coef']:.3f} -> approx {st['pct_change_approx']:.2f}% change"
            if sig and st['coef'] > 0:
                verdict = "Supports hypothesis: more feminine names associated with higher human cost (statistically significant)."
            elif sig and st['coef'] < 0:
                verdict = "Contradicts hypothesis: more feminine names associated with LOWER human cost (statistically significant)."
            else:
                verdict = "No strong evidence: association is not statistically significant at p<0.05."
            final_text = (
                f"Primary model used for verdict: {mname}. Variable NameFemScore_c has a {sign} coefficient. "
                f"{effect_desc}. p-value={st['p_value']:.3g}. {verdict}"
            )
        elif 'FemaleName' in vars_stats:
            st = vars_stats['FemaleName']
            sign = "positive" if st['coef'] > 0 else "negative"
            sig = st['significant_0.05']
            if st['is_count_model']:
                effect_desc = f"exp(coef)={st['exp_coef']:.3f} (95%CI [{st['exp_ci_95'][0]:.3f}, {st['exp_ci_95'][1]:.3f}])"
            else:
                effect_desc = f"coef={st['coef']:.3f} -> approx {st['pct_change_approx']:.2f}% change"
            if sig and st['coef'] > 0:
                verdict = "Supports hypothesis (female-named storms have higher human cost)."
            elif sig and st['coef'] < 0:
                verdict = "Contradicts hypothesis (female-named storms have lower human cost)."
            else:
                verdict = "No strong evidence (not statistically significant)."
            final_text = (
                f"Primary model used for verdict: {mname}. Variable FemaleName has a {sign} coefficient. "
                f"{effect_desc}. p-value={st['p_value']:.3g}. {verdict}"
            )
        else:
            final_text = "None of the variables of interest were found in the chosen primary model; cannot form verdict."

    desc_lines.append("")  # blank line
    desc_lines.append("Final verdict summary:")
    desc_lines.append(final_text)

    out['description'] = "\n".join(desc_lines)
    return out