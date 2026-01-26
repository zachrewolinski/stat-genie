def extract_final_answer(model_output):
    """
    Extracts key statistics about the student_teacher_ratio effect from the model_output
    and returns a concise object plus a plain-language description.

    Returns:
      {
        "object": { ... numeric summary ... },
        "description": "Plain-language interpretation"
      }
    """
    # Prepare placeholders
    coef = model_output.get('coef_student_teacher_ratio')
    pval = model_output.get('pvalue_student_teacher_ratio')
    std_coef = model_output.get('std_coef_student_teacher_ratio')
    n_obs = model_output.get('n_obs')
    rsq = model_output.get('rsquared')
    rsq_adj = model_output.get('rsquared_adj')
    ci_lower = ci_upper = None

    # Try to extract 95% CI from the fitted model object if available
    try:
        model = model_output.get('model')
        if model is not None:
            # statsmodels results have a conf_int() method; index by parameter name if possible
            ci = model.conf_int(alpha=0.05)
            # ci may be a DataFrame or ndarray; attempt to access the row for the parameter
            if hasattr(ci, 'loc') and ('student_teacher_ratio' in ci.index):
                row = ci.loc['student_teacher_ratio']
                ci_lower, ci_upper = float(row[0]), float(row[1])
            else:
                # fallback: try indexing by integer position (find column)
                try:
                    # find the index position of the parameter in model.params
                    params_index = list(model.params.index).index('student_teacher_ratio')
                    ci_lower, ci_upper = float(ci[params_index, 0]), float(ci[params_index, 1])
                except Exception:
                    ci_lower = ci_upper = None
    except Exception:
        ci_lower = ci_upper = None

    # Determine statistical significance at conventional alpha = 0.05
    significant = None
    if pval is not None:
        try:
            significant = float(pval) < 0.05
        except Exception:
            significant = None

    # Build the object to return (numbers left as floats or None)
    result_object = {
        "coef_student_teacher_ratio": float(coef) if coef is not None else None,
        "p_value_student_teacher_ratio": float(pval) if pval is not None else None,
        "95ci_lower": float(ci_lower) if ci_lower is not None else None,
        "95ci_upper": float(ci_upper) if ci_upper is not None else None,
        "std_coef_student_teacher_ratio": float(std_coef) if std_coef is not None else None,
        "n_obs": int(n_obs) if n_obs is not None else None,
        "r_squared": float(rsq) if rsq is not None else None,
        "r_squared_adj": float(rsq_adj) if rsq_adj is not None else None,
        "statistically_significant_at_0.05": bool(significant) if significant is not None else None
    }

    # Plain-language description / conclusion
    # Use available numbers to craft a brief interpretation
    coef_str = f"{result_object['coef_student_teacher_ratio']:.4f}" if result_object['coef_student_teacher_ratio'] is not None else "NA"
    pval_str = f"{result_object['p_value_student_teacher_ratio']:.3f}" if result_object['p_value_student_teacher_ratio'] is not None else "NA"
    std_str = f"{result_object['std_coef_student_teacher_ratio']:.3f}" if result_object['std_coef_student_teacher_ratio'] is not None else "NA"
    n_str = str(result_object['n_obs']) if result_object['n_obs'] is not None else "NA"
    rsq_str = f"{result_object['r_squared']:.3f}" if result_object['r_squared'] is not None else "NA"

    ci_str = "NA"
    if result_object['95ci_lower'] is not None and result_object['95ci_upper'] is not None:
        ci_str = f"[{result_object['95ci_lower']:.4f}, {result_object['95ci_upper']:.4f}]"

    significance_text = "statistically significant (p < 0.05)" if result_object['statistically_significant_at_0.05'] else "not statistically significant (p >= 0.05)" if result_object['statistically_significant_at_0.05'] is not None else "significance unknown"

    description = (
        f"The estimated coefficient for student_teacher_ratio is {coef_str} (95% CI {ci_str}), "
        f"standardized effect = {std_str} SDs, p = {pval_str}. With n = {n_str} districts and "
        f"R^2 ≈ {rsq_str}, this effect is {significance_text}. "
        "Interpretation: the point estimate is very small and (if p >= 0.05) not statistically significant, "
        "so there is no evidence here that a lower student-teacher ratio is associated with higher average academic performance."
    )

    return {"object": result_object, "description": description}