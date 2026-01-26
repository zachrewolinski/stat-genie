def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of instructor 'beauty' on student evaluations
    from the provided model_output dictionary.

    Returns a dictionary with:
      - "object": dict containing numeric results (coefficients, SE, p-values, 95% CI when available)
      - "description": a brief plain-language interpretation of the results in the task context
    """
    import numpy as np

    # Prepare result containers
    obj = {'ols': None, 'mixedlm': None, 'conclusion': None}
    desc_parts = []

    # Helper to safely pull numeric values and coerce to float/None
    def _get(d, k):
        try:
            v = d.get(k)
        except Exception:
            v = None
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    # 1) Extract from summary_beauty_ols if present
    sb_ols = model_output.get('summary_beauty_ols')
    ols_res = model_output.get('ols_result')
    if sb_ols is not None:
        coef = _get(sb_ols, 'coef_beauty')
        se = _get(sb_ols, 'se_beauty')
        pval = _get(sb_ols, 'pval_beauty')
        coef_sq = _get(sb_ols, 'coef_beauty_sq')
        se_sq = _get(sb_ols, 'se_beauty_sq')
        pval_sq = _get(sb_ols, 'pval_beauty_sq')

        # Try to get a 95% CI for beauty from the fitted ols_result if available
        ci_lower, ci_upper = (None, None)
        try:
            if ols_res is not None and hasattr(ols_res, 'conf_int'):
                ci = ols_res.conf_int()  # may be ndarray or DataFrame
                # find index of 'beauty' in params index if needed
                try:
                    # If conf_int returns a DataFrame
                    if hasattr(ci, 'loc'):
                        row = ci.loc['beauty']
                        ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
                    else:
                        # numpy array: get position from params index
                        params_index = list(ols_res.params.index)
                        pos = params_index.index('beauty')
                        ci_lower, ci_upper = float(ci[pos, 0]), float(ci[pos, 1])
                except Exception:
                    ci_lower, ci_upper = (None, None)
        except Exception:
            ci_lower, ci_upper = (None, None)

        obj['ols'] = {
            'coef_beauty': coef,
            'se_beauty': se,
            'pval_beauty': pval,
            '95ci_beauty': (ci_lower, ci_upper),
            'coef_beauty_sq': coef_sq,
            'se_beauty_sq': se_sq,
            'pval_beauty_sq': pval_sq
        }

        # Interpret magnitude relative to the evaluation scale (1-5 => range = 4)
        magnitude_note = None
        if coef is not None:
            # effect size on 1-5 scale per 1-unit change in mean-centered beauty
            pct_of_scale = (coef / 4.0) * 100.0
            magnitude_note = f"OLS: a one-unit increase in mean-centered beauty is associated with a {coef:.3f} point increase in the evaluation (≈{pct_of_scale:.1f}% of the 4-point 1-to-5 scale)."
        else:
            magnitude_note = "OLS: coefficient not available."

        significance_note = None
        if pval is not None:
            significance_note = ("statistically significant (p = {:.3g})".format(pval)
                                 if pval < 0.05 else "not statistically significant (p = {:.3g})".format(pval))
        else:
            significance_note = "p-value not available."

        desc_parts.append(magnitude_note + " " + significance_note)

        # Quadratic term interpretation
        if coef_sq is not None:
            desc_parts.append("Quadratic term (beauty_sq): coef = {:+.3f}, p = {:.3g}.".format(
                coef_sq, pval_sq if pval_sq is not None else np.nan))
        else:
            desc_parts.append("Quadratic term not available in OLS summary.")

    else:
        desc_parts.append("No OLS beauty summary found in model_output.")

    # 2) Extract from mixedlm result if present
    mixed_res = model_output.get('mixedlm_result')
    sb_mix = model_output.get('summary_beauty_mixed')
    if sb_mix is not None:
        coef_m = _get(sb_mix, 'coef_beauty')
        se_m = _get(sb_mix, 'se_beauty')
        pval_m = _get(sb_mix, 'pval_beauty')
        coef_m_sq = _get(sb_mix, 'coef_beauty_sq')
        se_m_sq = _get(sb_mix, 'se_beauty_sq')
        pval_m_sq = _get(sb_mix, 'pval_beauty_sq')

        obj['mixedlm'] = {
            'coef_beauty': coef_m,
            'se_beauty': se_m,
            'pval_beauty': pval_m,
            'coef_beauty_sq': coef_m_sq,
            'se_beauty_sq': se_m_sq,
            'pval_beauty_sq': pval_m_sq
        }

        # Note if mixed model estimates appear invalid/NA
        if coef_m is None or (se_m is None or np.isnan(se_m)):
            desc_parts.append("Mixed-effects model provided but standard errors/p-values for beauty are missing or invalid; rely on OLS cluster results instead.")
        else:
            desc_parts.append("Mixed-effects model: beauty coef = {:+.3f}, SE = {}, p = {}.".format(
                coef_m if coef_m is not None else np.nan,
                "{:.3f}".format(se_m) if se_m is not None else "NA",
                "{:.3g}".format(pval_m) if pval_m is not None else "NA"
            ))
    else:
        desc_parts.append("No mixed-effects beauty summary found in model_output.")

    # Final concise conclusion based on OLS (preferred because mixed had invalid SEs)
    final_conclusion = "Based primarily on the OLS model with clustered SEs: "
    if obj['ols'] is not None and obj['ols']['coef_beauty'] is not None:
        coef = obj['ols']['coef_beauty']
        pval = obj['ols']['pval_beauty']
        if pval is not None and pval < 0.05:
            final_conclusion += ("There is evidence that higher perceived instructor attractiveness is associated with higher student evaluation scores. "
                                 f"The estimated effect is +{coef:.3f} evaluation points per 1-unit increase in mean-centered beauty (p = {pval:.3g}).")
        else:
            final_conclusion += ("No statistically reliable evidence that attractiveness affects evaluations (coef = {:+.3f}, p = {:+.3g})."
                                 .format(coef, pval if pval is not None else np.nan))
    else:
        final_conclusion += "Insufficient information to draw a conclusion."

    obj['conclusion'] = final_conclusion
    description = " ".join(desc_parts) + " " + final_conclusion

    return {"object": obj, "description": description}