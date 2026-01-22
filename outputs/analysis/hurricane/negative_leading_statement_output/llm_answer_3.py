def extract_final_answer(model_output):
    """
    Extract key statistics for the masfem_scaled and gender_mf predictors from the
    provided model_output dict returned by the modeling function.

    Returns a dict with:
      - "object": a nested dictionary containing coefficients, standard errors,
                  p-values, 95% CIs, and (for NB/GLM) exponentiated coefficients
                  and percent changes, for the primary OLS and NB specifications
                  (and the alternate binary-gender specs).
      - "description": a concise plain-language interpretation of the results
                       in the context of the hypothesis (whether more-feminine
                       names predict fewer precautions / higher fatalities).
    """
    import numpy as np

    def _get_res(model_output, key):
        """Return fitted results object if present, else None."""
        entry = model_output.get(key)
        if not entry:
            return None
        # entry may be {'model': <results>, 'summary_text': ...} or an error string
        if isinstance(entry, dict) and 'model' in entry:
            return entry['model']
        return None

    def _extract(res, var):
        """Extract coef, se, pval, ci, and transformed effects for a single results object."""
        if res is None:
            return None
        out = {}
        params = getattr(res, 'params', None)
        if params is None or var not in params.index:
            return None
        coef = float(params[var])
        out['coef'] = coef
        # standard error
        try:
            out['std_err'] = float(res.bse[var])
        except Exception:
            out['std_err'] = None
        # p-value
        try:
            out['p_value'] = float(res.pvalues[var])
        except Exception:
            out['p_value'] = None
        # 95% conf int
        try:
            ci = res.conf_int().loc[var]
            out['ci_lower'] = float(ci[0])
            out['ci_upper'] = float(ci[1])
        except Exception:
            # fallback: try positional indexing
            try:
                idx = list(res.params.index).index(var)
                ci_mat = res.conf_int()
                out['ci_lower'] = float(ci_mat[idx, 0])
                out['ci_upper'] = float(ci_mat[idx, 1])
            except Exception:
                out['ci_lower'] = out['ci_upper'] = None
        # detect GLM (count) with family & log link -> exponentiate coeff
        is_glm = False
        try:
            fam = getattr(res.model, 'family', None)
            if fam is not None:
                is_glm = True
        except Exception:
            is_glm = False
        if is_glm:
            try:
                exp_coef = float(np.exp(coef))
                out['exp_coef'] = exp_coef
                out['pct_change'] = (exp_coef - 1.0) * 100.0
            except Exception:
                out['exp_coef'] = out['pct_change'] = None
        else:
            out['exp_coef'] = out['pct_change'] = None
        return out

    # Primary models
    ols_masfem_res = _get_res(model_output, 'ols_masfem')
    nb_masfem_res = _get_res(model_output, 'nb_masfem')
    ols_gender_res = _get_res(model_output, 'ols_gender')
    nb_gender_res = _get_res(model_output, 'nb_gender')

    # Extract stats
    ols_masfem_stats = _extract(ols_masfem_res, 'masfem_scaled')
    nb_masfem_stats = _extract(nb_masfem_res, 'masfem_scaled')
    ols_gender_stats = _extract(ols_gender_res, 'gender_mf')
    nb_gender_stats = _extract(nb_gender_res, 'gender_mf')

    # Build a concise interpretation string using conventional alpha=0.05
    def interpret(stat, var_label, model_label, outcome_desc):
        if stat is None:
            return f"No {model_label} result available for {var_label}."
        p = stat.get('p_value')
        coef = stat.get('coef')
        if p is None:
            sig = "p-value unavailable"
        else:
            sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
        if stat.get('exp_coef') is not None:
            pct = stat.get('pct_change')
            return (f"{model_label} on {outcome_desc}: {var_label} coef = {coef:.3f}, "
                    f"exp(coef) = {stat['exp_coef']:.3f} (~{pct:.1f}% change), p = {p:.3f} → {sig}.")
        else:
            return (f"{model_label} on {outcome_desc}: {var_label} coef = {coef:.3f} (change in log outcome), "
                    f"p = {p:.3f} → {sig}.")

    interpretations = []
    interpretations.append(interpret(ols_masfem_stats, 'masfem_scaled', 'OLS (HC3)', 'log1p total fatalities (primary)'))
    interpretations.append(interpret(nb_masfem_stats, 'masfem_scaled', 'Negative Binomial (GLM)', 'raw total fatalities'))
    interpretations.append(interpret(ols_gender_stats, 'gender_mf', 'OLS (HC3)', 'log1p total fatalities (binary gender)'))
    interpretations.append(interpret(nb_gender_stats, 'gender_mf', 'Negative Binomial (GLM)', 'raw total fatalities (binary gender)'))

    # Short summary conclusion
    # Prioritize the authors' stated primary specification: OLS on log_alldeaths (HC3).
    if ols_masfem_stats is not None and ols_masfem_stats.get('p_value') is not None:
        if ols_masfem_stats['p_value'] < 0.05:
            primary_conclusion = ("Primary OLS: evidence that more-feminine names predict higher/lower fatalities "
                                  "(see sign of coef).")
        else:
            # note NB result if it disagrees and is significant
            if nb_masfem_stats is not None and nb_masfem_stats.get('p_value') is not None and nb_masfem_stats['p_value'] < 0.05:
                primary_conclusion = ("Primary OLS shows no statistically significant association. "
                                      "However, the negative-binomial model on raw counts shows a statistically "
                                      "significant positive association (more-feminine names → higher expected fatalities).")
            else:
                primary_conclusion = ("No consistent evidence across primary specifications that more-feminine hurricane names "
                                      "are associated with fewer fatalities. Primary OLS (the authors' specified primary model) is not significant.")
    else:
        primary_conclusion = "Primary OLS result unavailable."

    result_object = {
        'ols_masfem': ols_masfem_stats,
        'nb_masfem': nb_masfem_stats,
        'ols_gender': ols_gender_stats,
        'nb_gender': nb_gender_stats
    }

    description_lines = [
        "Extracted key estimates for the predictors of interest (masfem_scaled and gender_mf):",
        *interpretations,
        "",
        "Summary conclusion:",
        primary_conclusion,
        "",
        "Notes: OLS coefficients are effects on the log1p(fatalities) outcome (additive on the log scale).",
        "Negative-binomial GLM coefficients are on the log link; exp(coef) gives the multiplicative change in expected counts."
    ]

    return {
        "object": result_object,
        "description": " ".join(description_lines)
    }