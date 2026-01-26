def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, 95% CI, effect per 10-unit change,
    sample size and R-squared for StudentTeacherRatio from models in model_output.
    
    Returns a dict with:
      - "object": dict of statistics for each model key in model_output (e.g., 'main','read','math')
      - "description": human-readable interpretation focusing on the main model
    """
    def get_stats(res):
        if res is None:
            return None
        var = 'StudentTeacherRatio'
        # Ensure variable present
        try:
            coef = float(res.params[var])
        except Exception:
            return None
        se = float(res.bse[var])
        t = float(res.tvalues[var])
        p = float(res.pvalues[var])
        ci_lower, ci_upper = map(float, res.conf_int(alpha=0.05).loc[var].tolist())
        effect_per_10 = coef * 10.0
        # nobs may be float; convert to int if appropriate
        try:
            n = int(res.nobs)
        except Exception:
            n = float(res.nobs)
        # R-squared (available for OLS)
        try:
            r2 = float(res.rsquared)
        except Exception:
            r2 = None
        return {
            'coef': coef,
            'se': se,
            't': t,
            'p': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_per_10': effect_per_10,
            'n': n,
            'r2': r2
        }

    # Extract stats for each model in the output
    stats = {}
    for key, res in model_output.items():
        stats[key] = get_stats(res)

    # Build interpretation focused on the main model
    main_stats = stats.get('main')
    if main_stats is None:
        description = "The main model does not contain an estimated coefficient for StudentTeacherRatio."
    else:
        coef = main_stats['coef']
        p = main_stats['p']
        ci_l = main_stats['ci_lower']
        ci_u = main_stats['ci_upper']
        se = main_stats['se']
        n = main_stats['n']
        r2 = main_stats['r2']
        # Directional interpretation: higher StudentTeacherRatio => more students per teacher.
        if coef < 0:
            direction = "negative"
            implied = "A lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore."
        elif coef > 0:
            direction = "positive"
            implied = "A lower student-teacher ratio is associated with lower AvgScore (opposite of the hypothesis)."
        else:
            direction = "zero"
            implied = "No association between student-teacher ratio and AvgScore."

        sig_text = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"

        description = (
            f"Main model estimate for StudentTeacherRatio: coef = {coef:.3f}, SE = {se:.3f}, "
            f"95% CI = [{ci_l:.3f}, {ci_u:.3f}], p = {p:.3f}. This coefficient is {direction} and {sig_text}. "
            f"{implied} Numerically, a decrease of 1 student per teacher is associated with a change in AvgScore of "
            f"{(-coef):.3f} points (i.e., {-coef:.3f} point increase if coef < 0). Effect for a 10-student decrease: "
            f"{(-main_stats['effect_per_10']):.3f} points. Model sample size n = {n}, R-squared = {r2:.3f}."
        )

    return {'object': stats, 'description': description}