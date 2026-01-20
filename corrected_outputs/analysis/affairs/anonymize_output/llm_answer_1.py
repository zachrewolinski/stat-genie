def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of 'Children' on extramarital affairs
    from the provided model_output dict containing fitted models.

    Returns a dict with:
      - "object": a dictionary with extracted numeric summaries for both the
                  inflation count model (ZINB or ZIP when available) and the
                  OLS robustness model.
      - "description": a concise interpretation answering whether having
                       children decreases engagement in extramarital affairs,
                       reporting effect sizes, CIs, and p-values.

    The function handles:
      - ZeroInflatedNegativeBinomial ('zinb_res') or fallback ZeroInflatedPoisson
        ('zip_res_fallback') if present in model_output.
      - OLS results under key 'ols_res'.
    """
    import numpy as np
    from scipy.stats import norm

    def find_param_name(params_index, target):
        """
        Find a parameter name in params_index corresponding to target (e.g., 'Children' or 'Children_Female')
        Prefer exact matches; otherwise accept names that contain target but are not inflation params.
        """
        # Try exact match
        if target in params_index:
            return target
        # Otherwise search for non-inflate names containing target
        for name in params_index:
            lname = str(name)
            if 'inflate' in lname.lower():
                continue
            if target.lower() in lname.lower():
                return name
        # As last resort, accept any name containing target (even if inflate)
        for name in params_index:
            if target.lower() in str(name).lower():
                return name
        return None

    def extract_from_countlike(res):
        """
        Given a statsmodels result object from a zero-inflated model,
        extract Children and Children_Female coefficients for the count process,
        compute combined effect for females, and return numeric summaries.
        """
        out = {
            'model_type': type(res).__name__,
            'children_coef': None,
            'children_se': None,
            'children_pval': None,
            'children_ci95': None,
            'children_irratio': None,  # exp(coef)
            'children_irratio_ci95': None,
            'children_female_coef': None,
            'female_combined_coef': None,
            'female_combined_se': None,
            'female_combined_pval': None,
            'female_irratio': None,
            'female_irratio_ci95': None
        }

        # Access params, pvalues, cov_params
        params = getattr(res, 'params', None)
        pvalues = getattr(res, 'pvalues', None)
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        if params is None:
            return out

        index = list(params.index)

        # Locate the count-model parameter names (non-inflation)
        child_name = find_param_name(index, 'Children')
        child_female_name = find_param_name(index, 'Children_Female')

        # If found, extract
        if child_name is not None:
            coef = float(params[child_name])
            se = float(np.sqrt(cov.loc[child_name, child_name])) if cov is not None and child_name in cov.index else (float(getattr(res, 'bse', {}).get(child_name, np.nan)) if hasattr(res, 'bse') else None)
            pval = float(pvalues[child_name]) if pvalues is not None and child_name in pvalues.index else None
            ci_low = coef - 1.96 * se if se is not None else None
            ci_high = coef + 1.96 * se if se is not None else None

            out.update({
                'children_coef': coef,
                'children_se': se,
                'children_pval': pval,
                'children_ci95': (ci_low, ci_high) if (ci_low is not None and ci_high is not None) else None,
                'children_irratio': float(np.exp(coef)) if coef is not None else None,
                'children_irratio_ci95': (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else None
            })

        # Extract children_female interaction if present (it will be added to children for females)
        if child_female_name is not None and child_name is not None:
            coef_int = float(params[child_female_name])
            out['children_female_coef'] = coef_int

            # Combined effect for females = coef_child + coef_int
            combined = out['children_coef'] + coef_int
            # Compute SE of combined using covariance if available
            if cov is not None and child_name in cov.index and child_female_name in cov.index:
                var_comb = cov.loc[child_name, child_name] + cov.loc[child_female_name, child_female_name] + 2 * cov.loc[child_name, child_female_name]
                se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else float(np.nan)
                z = combined / se_comb if se_comb is not None and se_comb != 0 else None
                p_comb = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
                ci_low = combined - 1.96 * se_comb
                ci_high = combined + 1.96 * se_comb
            else:
                # If covariance unavailable, try to use bse as approximation and assume independence (conservative)
                se_child = out['children_se']
                se_int = float(np.sqrt(cov.loc[child_female_name, child_female_name])) if cov is not None and child_female_name in cov.index else (float(getattr(res, 'bse', {}).get(child_female_name, np.nan)) if hasattr(res, 'bse') else None)
                if se_child is not None and se_int is not None:
                    se_comb = float(np.sqrt(se_child ** 2 + se_int ** 2))
                    z = combined / se_comb if se_comb != 0 else None
                    p_comb = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
                    ci_low = combined - 1.96 * se_comb
                    ci_high = combined + 1.96 * se_comb
                else:
                    se_comb = None
                    p_comb = None
                    ci_low = None
                    ci_high = None

            out.update({
                'female_combined_coef': combined,
                'female_combined_se': se_comb,
                'female_combined_pval': p_comb,
                'female_irratio': float(np.exp(combined)) if combined is not None else None,
                'female_irratio_ci95': (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else None
            })
        else:
            # No interaction term: the children effect applies to both sexes equally
            if out['children_coef'] is not None:
                out.update({
                    'female_combined_coef': out['children_coef'],
                    'female_combined_se': out['children_se'],
                    'female_combined_pval': out['children_pval'],
                    'female_irratio': out['children_irratio'],
                    'female_irratio_ci95': out['children_irratio_ci95']
                })

        return out

    def extract_from_ols(res):
        """
        Extract Children and Children_Female from OLS result (on log1p outcome).
        Return coef, se, pval, CI, and combined female effect similar to above.
        """
        out = {
            'model_type': 'OLS_on_log1p',
            'children_coef': None,
            'children_se': None,
            'children_pval': None,
            'children_ci95': None,
            'female_combined_coef': None,
            'female_combined_se': None,
            'female_combined_pval': None,
            'female_ci95': None
        }
        params = getattr(res, 'params', None)
        pvalues = getattr(res, 'pvalues', None)
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        if params is None:
            return out

        index = list(params.index)
        child_name = find_param_name(index, 'Children')
        child_female_name = find_param_name(index, 'Children_Female')

        if child_name is not None:
            coef = float(params[child_name])
            se = float(np.sqrt(cov.loc[child_name, child_name])) if cov is not None and child_name in cov.index else float(getattr(res, 'bse', {}).get(child_name, np.nan)) if hasattr(res, 'bse') else None
            pval = float(pvalues[child_name]) if pvalues is not None and child_name in pvalues.index else None
            ci_low = coef - 1.96 * se if se is not None else None
            ci_high = coef + 1.96 * se if se is not None else None
            out.update({
                'children_coef': coef,
                'children_se': se,
                'children_pval': pval,
                'children_ci95': (ci_low, ci_high) if ci_low is not None else None
            })

        if child_female_name is not None and child_name is not None:
            coef_int = float(params[child_female_name])
            combined = out['children_coef'] + coef_int
            # compute se for combined
            if cov is not None and child_name in cov.index and child_female_name in cov.index:
                var_comb = cov.loc[child_name, child_name] + cov.loc[child_female_name, child_female_name] + 2 * cov.loc[child_name, child_female_name]
                se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else float(np.nan)
                z = combined / se_comb if se_comb is not None and se_comb != 0 else None
                p_comb = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
                ci_low = combined - 1.96 * se_comb
                ci_high = combined + 1.96 * se_comb
            else:
                se_child = out['children_se']
                se_int = float(np.sqrt(cov.loc[child_female_name, child_female_name])) if cov is not None and child_female_name in cov.index else (float(getattr(res, 'bse', {}).get(child_female_name, np.nan)) if hasattr(res, 'bse') else None)
                if se_child is not None and se_int is not None:
                    se_comb = float(np.sqrt(se_child ** 2 + se_int ** 2))
                    z = combined / se_comb if se_comb != 0 else None
                    p_comb = float(2 * (1 - norm.cdf(abs(z)))) if z is not None else None
                    ci_low = combined - 1.96 * se_comb
                    ci_high = combined + 1.96 * se_comb
                else:
                    se_comb = None
                    p_comb = None
                    ci_low = None
                    ci_high = None

            out.update({
                'female_combined_coef': combined,
                'female_combined_se': se_comb,
                'female_combined_pval': p_comb,
                'female_ci95': (ci_low, ci_high) if ci_low is not None else None
            })
        else:
            # no interaction: same effect for females
            if out['children_coef'] is not None:
                out.update({
                    'female_combined_coef': out['children_coef'],
                    'female_combined_se': out['children_se'],
                    'female_combined_pval': out['children_pval'],
                    'female_ci95': out['children_ci95']
                })

        return out

    results_summary = {}
    descriptions = []

    # 1) Extract from inflation model (ZINB or ZIP)
    infl_model = None
    if 'zinb_res' in model_output and model_output['zinb_res'] is not None:
        infl_model = model_output['zinb_res']
    elif 'zip_res_fallback' in model_output and model_output['zip_res_fallback'] is not None:
        infl_model = model_output['zip_res_fallback']

    if infl_model is not None:
        try:
            zinb_stats = extract_from_countlike(infl_model)
            results_summary['inflation_count_model'] = zinb_stats

            # Interpret effect: for count models, coefficient = log change in expected count;
            # exp(coef) = multiplicative change (IRR).
            # Provide concise interpretation for males (baseline) and females (combined).
            # Determine significance using p-values where available.
            def interpret_count(stats):
                s = []
                # Males (baseline)
                if stats['children_coef'] is not None:
                    irr = stats['children_irratio']
                    p = stats['children_pval']
                    ci = stats['children_irratio_ci95']
                    s.append(f"Count-model (men/baseline): Children coef = {stats['children_coef']:.3f}, IRR = {irr:.3f}" +
                             (f", 95% CI IRR = ({ci[0]:.3f}, {ci[1]:.3f})" if ci is not None else "") +
                             (f", p = {p:.3f}" if p is not None else ""))
                # Females (combined)
                if stats['female_combined_coef'] is not None:
                    irr_f = stats['female_irratio']
                    p_f = stats['female_combined_pval']
                    ci_f = stats['female_irratio_ci95']
                    s.append(f"Count-model (females): combined Children effect coef = {stats['female_combined_coef']:.3f}, IRR = {irr_f:.3f}" +
                             (f", 95% CI IRR = ({ci_f[0]:.3f}, {ci_f[1]:.3f})" if ci_f is not None else "") +
                             (f", p = {p_f:.3f}" if p_f is not None else ""))
                return " | ".join(s)

            descriptions.append(interpret_count(zinb_stats))
        except Exception as e:
            results_summary['inflation_count_model'] = {'error': str(e)}
            descriptions.append("Inflation count model extraction failed: " + str(e))
    else:
        results_summary['inflation_count_model'] = None
        descriptions.append("No zero-inflated count model found in model_output.")

    # 2) Extract from OLS robustness model
    ols_stat = None
    if 'ols_res' in model_output and model_output['ols_res'] is not None:
        try:
            ols_stat = extract_from_ols(model_output['ols_res'])
            results_summary['ols_model'] = ols_stat

            # Interpret OLS effect: on log1p(Affairs). Small coefficients approximate proportional changes.
            def interpret_ols(s):
                parts = []
                if s['children_coef'] is not None:
                    parts.append(f"OLS on log1p: Children coef = {s['children_coef']:.3f}, 95% CI = ({s['children_ci95'][0]:.3f}, {s['children_ci95'][1]:.3f})" if s['children_ci95'] is not None else f"OLS on log1p: Children coef = {s['children_coef']:.3f}")
                    if s['children_pval'] is not None:
                        parts[-1] += f", p = {s['children_pval']:.3f}"
                if s['female_combined_coef'] is not None:
                    parts.append(f"OLS on log1p (females combined): coef = {s['female_combined_coef']:.3f}, 95% CI = ({s['female_ci95'][0]:.3f}, {s['female_ci95'][1]:.3f})" if s['female_ci95'] is not None else f"OLS on log1p (females): coef = {s['female_combined_coef']:.3f}")
                    if s['female_combined_pval'] is not None:
                        parts[-1] += f", p = {s['female_combined_pval']:.3f}"
                return " | ".join(parts)

            descriptions.append(interpret_ols(ols_stat))
            results_summary['interpretation'] = " | ".join(descriptions)
        except Exception as e:
            results_summary['ols_model'] = {'error': str(e)}
            descriptions.append("OLS extraction failed: " + str(e))
            results_summary['interpretation'] = " | ".join(descriptions)
    else:
        results_summary['ols_model'] = None
        descriptions.append("No OLS model found in model_output.")
        results_summary['interpretation'] = " | ".join(descriptions)

    # Final concise statement answering the question, using available stats:
    final_lines = []
    # Use inflation count model first if available
    if 'inflation_count_model' in results_summary and results_summary['inflation_count_model']:
        s = results_summary['inflation_count_model']
        if isinstance(s, dict) and s.get('children_coef') is not None:
            # Interpret direction and significance for baseline (men)
            coef = s['children_coef']
            irr = s['children_irratio']
            p = s['children_pval']
            if p is not None and p < 0.05:
                sig = "statistically significant"
            else:
                sig = "not statistically significant"
            direction = "decrease" if irr < 1 else ("increase" if irr > 1 else "no change")
            final_lines.append(f"Zero-inflated count model: For the baseline (men), presence of children is associated with a {100*(1-irr):.1f}% {direction} in expected affair counts (IRR={irr:.3f}, p={p:.3f}) — {sig}." if p is not None else f"Zero-inflated count model: For the baseline (men), IRR={irr:.3f}.")
            # Females
            if s.get('female_irratio') is not None:
                irr_f = s['female_irratio']
                p_f = s.get('female_combined_pval')
                if p_f is not None and p_f < 0.05:
                    sigf = "statistically significant"
                else:
                    sigf = "not statistically significant"
                directionf = "decrease" if irr_f < 1 else ("increase" if irr_f > 1 else "no change")
                final_lines.append(f"For females, presence of children is associated with a {100*(1-irr_f):.1f}% {directionf} in expected affair counts (IRR={irr_f:.3f}, p={p_f:.3f}) — {sigf}." if p_f is not None else f"For females, IRR={irr_f:.3f}.")
        else:
            final_lines.append("Zero-inflated count model did not yield extractable 'Children' estimates.")
    # Fall back to OLS summary if count model absent
    if ('inflation_count_model' not in results_summary or not results_summary['inflation_count_model']) and 'ols_model' in results_summary and results_summary['ols_model']:
        s = results_summary['ols_model']
        if s.get('children_coef') is not None:
            coef = s['children_coef']
            p = s['children_pval']
            effect_pct = (np.exp(coef) - 1) * 100 if coef is not None else None
            if p is not None and p < 0.05:
                sig = "statistically significant"
            else:
                sig = "not statistically significant"
            final_lines.append(f"OLS (log1p outcome): Children coef = {coef:.3f} (approx {effect_pct:.1f}% change), p = {p:.3f} — {sig}.")

    if not final_lines:
        final_lines = ["Could not extract interpretable estimates for 'Children' from provided models."]

    final_statement = " ".join(final_lines)

    return {
        "object": results_summary,
        "description": final_statement
    }