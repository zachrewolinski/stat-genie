def extract_final_answer(model_output):
    """
    Extract coefficient, standard error, t-stat, p-value, 95% CI, sample size, R^2,
    and produce a short interpretation for the StudentTeacherRatio coefficient
    from a statsmodels RegressionResultsWrapper.
    Returns: {'object': {...}, 'description': '...'}
    """
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        tvalues = model_output.tvalues
        ci = model_output.conf_int()  # DataFrame or ndarray with index matching params

        # Identify the parameter name for student-teacher ratio
        param_name = 'StudentTeacherRatio'
        if param_name not in params.index:
            # fallback: try to find a param containing both words
            candidates = [n for n in params.index if 'student' in n.lower() and 'teacher' in n.lower()]
            if candidates:
                param_name = candidates[0]
            else:
                raise KeyError("StudentTeacherRatio parameter not found in model output.")

        coef = float(params[param_name])
        std_err = float(bse[param_name]) if param_name in bse.index else None
        t_stat = float(tvalues[param_name]) if param_name in tvalues.index else None
        p_value = float(pvalues[param_name])
        ci_lower, ci_upper = (float(ci.loc[param_name][0]), float(ci.loc[param_name][1]))

        # Additional info
        n_obs = int(getattr(model_output, 'nobs', getattr(model_output.model, 'nobs', None) or len(model_output.model.endog)))
        r_squared = float(getattr(model_output, 'rsquared', float('nan')))

        significant = p_value < 0.05
        direction = 'negative' if coef < 0 else ('positive' if coef > 0 else 'zero')

        # Short verdict about whether a lower student-teacher ratio is associated with higher performance:
        if coef < 0 and significant:
            verdict = ("Yes — coefficient is negative and statistically significant: "
                       "lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore.")
        elif coef < 0 and not significant:
            verdict = ("Coefficient is negative but not statistically significant: "
                       "point estimate suggests lower ratio is associated with higher AvgScore, but evidence is weak.")
        elif coef > 0 and significant:
            verdict = ("No — coefficient is positive and statistically significant: "
                       "higher student-teacher ratio is associated with higher AvgScore (contrary to the hypothesis).")
        elif coef > 0 and not significant:
            verdict = ("Coefficient is positive but not statistically significant: "
                       "no reliable evidence that student-teacher ratio is associated with AvgScore.")
        else:
            verdict = "Coefficient is essentially zero; no association detected."

        output_object = {
            'param': param_name,
            'coef': coef,
            'std_err': std_err,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r_squared,
            'significant_at_0.05': bool(significant),
            'direction': direction,
            'verdict': verdict
        }

        description = (
            f"Parameter '{param_name}': coef={coef:.4f}, SE={std_err:.4f}, t={t_stat:.2f}, p={p_value:.3g}; "
            f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]. {verdict}"
        )

        return {'object': output_object, 'description': description}

    except Exception as e:
        return {
            'object': None,
            'description': f"Could not extract results for StudentTeacherRatio: {e}"
        }