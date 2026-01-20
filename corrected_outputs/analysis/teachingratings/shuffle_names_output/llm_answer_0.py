def extract_final_answer(model_output):
    """
    Extracts coefficient, robust SE, t-value, p-value, 95% CI, nobs, and R-squared
    for the beauty variables from the provided statsmodels results objects.

    Expects model_output to be a dict with keys:
      - 'continuous_beauty_model'  -> statsmodels RegressionResultsWrapper (Beauty_z)
      - 'binary_top_quartile_beauty_model' -> statsmodels RegressionResultsWrapper (BeautyHigh)

    Returns a dict with:
      - "object": nested dict with extracted numeric results for each model/variable
      - "description": brief textual interpretation of the estimates in context
    """
    results = {}
    def _extract_from_result(res, varname):
        # Default return if variable not present
        if varname not in getattr(res, "params", {}).index:
            return None

        # Basic stats
        coef = float(res.params[varname])
        se = float(res.bse[varname])
        tval = float(res.tvalues[varname])
        pval = float(res.pvalues[varname])

        # 95% confidence interval (handle different return types)
        try:
            ci = res.conf_int().loc[varname]
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        except Exception:
            # fallback if conf_int returns ndarray
            conf_arr = res.conf_int()
            try:
                idx = list(res.model.exog_names).index(varname)
                ci_lower = float(conf_arr[idx, 0])
                ci_upper = float(conf_arr[idx, 1])
            except Exception:
                ci_lower = None
                ci_upper = None

        # Sample size and R-squared (if available)
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None
        try:
            rsq = float(res.rsquared)
        except Exception:
            rsq = None

        return {
            "variable": varname,
            "coef": coef,
            "se": se,
            "t": tval,
            "p": pval,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "nobs": nobs,
            "r_squared": rsq
        }

    # Defensive extraction
    cont_res = model_output.get('continuous_beauty_model')
    bin_res = model_output.get('binary_top_quartile_beauty_model')

    cont_stats = None
    bin_stats = None
    if cont_res is not None:
        cont_stats = _extract_from_result(cont_res, 'Beauty_z')
        results['continuous_beauty_model'] = cont_stats
    else:
        results['continuous_beauty_model'] = None

    if bin_res is not None:
        bin_stats = _extract_from_result(bin_res, 'BeautyHigh')
        results['binary_top_quartile_beauty_model'] = bin_stats
    else:
        results['binary_top_quartile_beauty_model'] = None

    # Build a short human-readable description
    desc_lines = []
    if cont_stats is not None:
        desc_lines.append(
            "Continuous beauty (Beauty_z): coef = {coef:.3f}, SE = {se:.3f}, p = {p:.3g}, "
            "95% CI = [{low:.3f}, {high:.3f}]. "
            "Interpretation: a one standard-deviation increase in perceived beauty is associated "
            "with a change of {coef:.3f} points in course evaluation (1-5 scale)."
            .format(coef=cont_stats['coef'], se=cont_stats['se'], p=cont_stats['p'],
                    low=cont_stats['ci_lower_95'] if cont_stats['ci_lower_95'] is not None else float('nan'),
                    high=cont_stats['ci_upper_95'] if cont_stats['ci_upper_95'] is not None else float('nan'))
        )
        if cont_stats['p'] < 0.05:
            desc_lines.append("This effect is statistically significant at alpha=0.05.")
        else:
            desc_lines.append("This effect is not statistically significant at alpha=0.05.")
    else:
        desc_lines.append("Continuous beauty (Beauty_z) statistic not available in model output.")

    if bin_stats is not None:
        desc_lines.append(
            "Binary top-quartile beauty (BeautyHigh): coef = {coef:.3f}, SE = {se:.3f}, p = {p:.3g}, "
            "95% CI = [{low:.3f}, {high:.3f}]. "
            "Interpretation: being in the top quartile of perceived beauty is associated with a "
            "{coef:.3f} point difference in course evaluation compared to others."
            .format(coef=bin_stats['coef'], se=bin_stats['se'], p=bin_stats['p'],
                    low=bin_stats['ci_lower_95'] if bin_stats['ci_lower_95'] is not None else float('nan'),
                    high=bin_stats['ci_upper_95'] if bin_stats['ci_upper_95'] is not None else float('nan'))
        )
        if bin_stats['p'] < 0.05:
            desc_lines.append("This difference is statistically significant at alpha=0.05.")
        else:
            desc_lines.append("This difference is not statistically significant at alpha=0.05.")
    else:
        desc_lines.append("Binary beauty (BeautyHigh) statistic not available in model output.")

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }