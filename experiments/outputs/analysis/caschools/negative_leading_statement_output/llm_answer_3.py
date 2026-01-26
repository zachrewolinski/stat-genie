def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of StudentTeacherRatio on AvgScore from the
    model_output produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coefficients, p-values,
                  confidence intervals, marginal effect if quadratic term present)
      - "description": a short interpretation in context (yes/no answer + caveats)
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy import stats
    except Exception:
        # Fallback to approximate normal cdf using math.erf if scipy not available
        import math
        def _norm_cdf(x):
            return 0.5 * (1 + math.erf(x / sqrt(2)))
        stats = None

    out = {
        'coef_linear': None,
        'pval_linear': None,
        'ci_95_linear': None,
        'coef_quadratic': None,
        'pval_quadratic': None,
        'marginal_effect_at_mean': None,
        'se_marginal_effect': None,
        'ci_95_marginal_effect': None,
        'pval_marginal_effect': None
    }

    # Try to extract from full results object if present
    results = model_output.get('results', None)

    # Fallback values included by the original model function
    fallback_coef = model_output.get('coef_student_teacher_ratio', None)
    fallback_pval = model_output.get('pval_student_teacher_ratio', None)

    # Helper to safe-format CI given a 2D array or DataFrame from results.conf_int()
    def _get_conf_int_for_name(conf_int_obj, name, exog_names=None):
        try:
            # If conf_int_obj is a DataFrame or has .loc
            return tuple(conf_int_obj.loc[name].tolist())
        except Exception:
            # If it's an array-like with ordering exog_names
            if exog_names is None or conf_int_obj is None:
                return None
            try:
                idx = exog_names.index(name)
                return (float(conf_int_obj[idx, 0]), float(conf_int_obj[idx, 1]))
            except Exception:
                return None

    def _get_from_container(container, name, exog_names=None):
        """
        Safely extract a named value from container which may be:
        - pandas Series / DataFrame row (supports .get or indexing by name)
        - dict-like (supports get)
        - numpy.ndarray (then use exog_names to find index)
        Returns None if not found.
        """
        if container is None:
            return None
        # dict-like or pandas Series
        try:
            # dict
            if isinstance(container, dict):
                val = container.get(name, None)
                return float(val) if val is not None else None
            # pandas Series or anything with .get
            if hasattr(container, 'get'):
                try:
                    val = container.get(name, None)
                    if val is not None:
                        return float(val)
                except Exception:
                    pass
            # pandas Series indexing
            try:
                if hasattr(container, 'index') and name in container.index:
                    return float(container[name])
            except Exception:
                pass
        except Exception:
            pass

        # array-like fallback: require exog_names
        try:
            if exog_names is not None and hasattr(container, '__len__') and not hasattr(container, 'index'):
                idx = exog_names.index(name)
                return float(container[idx])
        except Exception:
            pass

        return None

    if results is not None:
        try:
            params = results.params
            pvalues = results.pvalues
            # try to get exog_names if available
            exog_names = None
            try:
                exog_names = list(results.model.exog_names)
            except Exception:
                try:
                    # fallback if exog_names is a plain list/tuple
                    exog_names = list(getattr(results.model, 'exog_names', None)) if getattr(results.model, 'exog_names', None) is not None else None
                except Exception:
                    exog_names = None
        except Exception:
            params = getattr(results, 'params', None)
            pvalues = getattr(results, 'pvalues', None)
            exog_names = None

        # Linear term: use helper to extract value regardless of container type
        coef_lin = _get_from_container(params, 'StudentTeacherRatio', exog_names)
        if coef_lin is not None:
            out['coef_linear'] = float(coef_lin)
        else:
            out['coef_linear'] = fallback_coef

        pval_lin = _get_from_container(pvalues, 'StudentTeacherRatio', exog_names)
        if pval_lin is not None:
            out['pval_linear'] = float(pval_lin)
        else:
            out['pval_linear'] = fallback_pval

        # 95% CI for linear term
        try:
            conf_int = results.conf_int()
            ci_lin = _get_conf_int_for_name(conf_int, 'StudentTeacherRatio', exog_names)
            out['ci_95_linear'] = ci_lin
        except Exception:
            out['ci_95_linear'] = None

        # Quadratic term (if present)
        coef_quad = _get_from_container(params, 'StudentTeacherRatio_sq', exog_names)
        if coef_quad is not None:
            out['coef_quadratic'] = float(coef_quad)
            pval_quad = _get_from_container(pvalues, 'StudentTeacherRatio_sq', exog_names)
            if pval_quad is not None:
                out['pval_quadratic'] = float(pval_quad)
            # Compute marginal effect at sample mean if data available
            try:
                # Try to get mean of StudentTeacherRatio from model data (formula model stores a DataFrame)
                mean_str = None
                # Option 1: results.model.data.frame (pandas DataFrame)
                df = getattr(results.model, 'data', None)
                if df is not None:
                    # statsmodels stores the original frame in .data.frame for formula models
                    frame = getattr(df, 'frame', None)
                    if frame is not None and 'StudentTeacherRatio' in frame.columns:
                        mean_str = float(frame['StudentTeacherRatio'].mean())
                # Option 2: use exog column mean (exog matches exog_names)
                if mean_str is None:
                    exog = getattr(results.model, 'exog', None)
                    if exog is not None and exog_names is not None and 'StudentTeacherRatio' in exog_names:
                        idx = exog_names.index('StudentTeacherRatio')
                        # exog might be 2D
                        arr = np.asarray(exog)
                        if arr.ndim == 2:
                            mean_str = float(arr[:, idx].mean())
                        else:
                            mean_str = float(arr[idx].mean())
                # If mean found, compute marginal effect and its SE using delta method
                if mean_str is not None:
                    b1 = _get_from_container(params, 'StudentTeacherRatio', exog_names)
                    b2 = _get_from_container(params, 'StudentTeacherRatio_sq', exog_names)
                    if b1 is None or b2 is None:
                        # cannot compute without both coefficients
                        pass
                    else:
                        b1 = float(b1)
                        b2 = float(b2)
                        me = b1 + 2.0 * b2 * mean_str
                        # variance of me = Var(b1) + (2*mean)^2 Var(b2) + 2*(2*mean)*Cov(b1,b2)
                        cov = results.cov_params()
                        # cov could be DataFrame or ndarray
                        try:
                            var_b1 = float(cov.loc['StudentTeacherRatio', 'StudentTeacherRatio'])
                            var_b2 = float(cov.loc['StudentTeacherRatio_sq', 'StudentTeacherRatio_sq'])
                            cov_b1b2 = float(cov.loc['StudentTeacherRatio', 'StudentTeacherRatio_sq'])
                        except Exception:
                            # assume ndarray with ordering exog_names
                            if exog_names is not None:
                                i = exog_names.index('StudentTeacherRatio')
                                j = exog_names.index('StudentTeacherRatio_sq')
                                var_b1 = float(cov[i, i])
                                var_b2 = float(cov[j, j])
                                cov_b1b2 = float(cov[i, j])
                            else:
                                raise
                        factor = 2.0 * mean_str
                        var_me = var_b1 + (factor ** 2) * var_b2 + 2.0 * factor * cov_b1b2
                        se_me = sqrt(var_me) if var_me >= 0 else float('nan')
                        # 95% CI and p-value
                        if se_me is not None and not np.isnan(se_me) and se_me != 0:
                            z = me / se_me
                        else:
                            z = float('nan')
                        if stats is not None:
                            p_me = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
                        else:
                            p_me = 2.0 * (1.0 - _norm_cdf(abs(z)))
                        ci_lo = me - 1.96 * se_me
                        ci_hi = me + 1.96 * se_me
                        out['marginal_effect_at_mean'] = float(me)
                        out['se_marginal_effect'] = float(se_me)
                        out['ci_95_marginal_effect'] = (float(ci_lo), float(ci_hi))
                        out['pval_marginal_effect'] = float(p_me)
                else:
                    # no mean available, cannot compute marginal effect
                    out['marginal_effect_at_mean'] = None
            except Exception:
                # If anything fails, leave marginal effect fields as None
                pass

        # If quadratic not present, we can still report linear estimates (already set)
    else:
        # No results object: fall back to the provided scalar outputs if available
        out['coef_linear'] = fallback_coef
        out['pval_linear'] = fallback_pval

    # Build a concise description / interpretation
    # Determine whether effect indicates that lower student-teacher ratio (fewer students per teacher)
    # is associated with higher AvgScore. Lower ratio => smaller StudentTeacherRatio value.
    # A negative coefficient implies higher ratio -> lower score, equivalently lower ratio -> higher score.
    desc_parts = []
    coef_lin = out['coef_linear']
    p_lin = out['pval_linear']

    if coef_lin is None:
        desc_parts.append("Could not extract coefficient for StudentTeacherRatio.")
    else:
        try:
            coef_lin_val = float(coef_lin)
            desc_parts.append(f"The estimated linear coefficient for StudentTeacherRatio is {coef_lin_val:.3f}.")
        except Exception:
            desc_parts.append(f"The estimated linear coefficient for StudentTeacherRatio is {coef_lin}.")

        if out['ci_95_linear'] is not None:
            ci = out['ci_95_linear']
            try:
                desc_parts.append(f"95% CI ≈ ({ci[0]:.3f}, {ci[1]:.3f}).")
            except Exception:
                desc_parts.append(f"95% CI ≈ {ci}.")
        if p_lin is not None:
            try:
                desc_parts.append(f"Two-sided p-value = {float(p_lin):.3f}.")
            except Exception:
                desc_parts.append(f"Two-sided p-value = {p_lin}.")

        # Interpret sign and significance
        try:
            coef_num = float(coef_lin)
            if coef_num < 0:
                sign_interp = "direction: lower student-teacher ratio (fewer students per teacher) is associated with higher average scores."
            elif coef_num > 0:
                sign_interp = "direction: lower student-teacher ratio is associated with lower average scores (unexpected sign)."
            else:
                sign_interp = "no linear association (coefficient is zero)."
        except Exception:
            sign_interp = "Could not interpret the sign of the coefficient."

        # significance
        sig_interp = ""
        if p_lin is not None:
            try:
                pval_num = float(p_lin)
                if pval_num < 0.05:
                    sig_interp = "This effect is statistically significant at the 5% level."
                else:
                    sig_interp = "This effect is NOT statistically significant at the 5% level."
            except Exception:
                sig_interp = ""
        desc_parts.append(sign_interp + (" " + sig_interp if sig_interp else ""))

    # If marginal effect from quadratic was computed, include it
    if out['marginal_effect_at_mean'] is not None:
        me = out['marginal_effect_at_mean']
        se_me = out['se_marginal_effect']
        ci_me = out['ci_95_marginal_effect']
        p_me = out['pval_marginal_effect']
        try:
            desc_parts.append(
                f"Because the model includes StudentTeacherRatio_sq, the marginal effect at the sample mean ratio is "
                f"{float(me):.3f} (SE={float(se_me):.3f}, 95% CI=({float(ci_me[0]):.3f}, {float(ci_me[1]):.3f}), p={float(p_me):.3f})."
            )
        except Exception:
            desc_parts.append(f"Marginal effect at the sample mean: {me}, SE={se_me}, CI={ci_me}, p={p_me}.")

        try:
            if float(me) < 0:
                desc_parts.append("This marginal effect also implies that, at the sample mean, reducing the student-teacher ratio is associated with higher scores.")
            else:
                desc_parts.append("This marginal effect implies the opposite at the sample mean.")
        except Exception:
            pass

        if p_me is not None:
            try:
                if float(p_me) < 0.05:
                    desc_parts.append("The marginal effect at the mean is statistically significant at conventional levels.")
                else:
                    desc_parts.append("The marginal effect at the mean is NOT statistically significant at conventional levels.")
            except Exception:
                pass

    description = " ".join(desc_parts)

    return {
        "object": out,
        "description": description
    }