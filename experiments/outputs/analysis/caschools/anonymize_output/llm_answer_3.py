def extract_final_answer(model_output):
    """
    Extracts key statistics about the student-teacher ratio effect from the provided
    model_output (expected to contain 'model_linear' and/or 'model_log' statsmodels results).
    Returns a dict with keys:
      - "object": dict of extracted stats for each specification and an overall conclusion
      - "description": plain-language interpretation of those statistics relative to the question:
          "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    import numpy as np

    results = {}
    stats = {}
    sig_flags = []

    specs = [
        ('model_linear', 'StudentTeacherRatio'),
        ('model_log', 'LogStudentTeacherRatio')
    ]

    for spec_name, varname in specs:
        if spec_name not in model_output:
            continue
        model = model_output[spec_name]
        # initialize with NaNs
        entry = {
            'coef': np.nan,
            'se': np.nan,
            't': np.nan,
            'p': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': None,
            'variable': varname,
            'specification': spec_name
        }
        try:
            params = getattr(model, 'params', {})
            bse = getattr(model, 'bse', {})
            tvals = getattr(model, 'tvalues', {})
            pvals = getattr(model, 'pvalues', {})
            ci = getattr(model, 'conf_int', lambda: None)()
            entry['coef'] = float(params.get(varname, np.nan))
            entry['se'] = float(bse.get(varname, np.nan))
            entry['t'] = float(tvals.get(varname, np.nan))
            entry['p'] = float(pvals.get(varname, np.nan))
            # confidence interval extraction (if available)
            if ci is not None:
                try:
                    # statsmodels returns DataFrame; use .loc if possible
                    if hasattr(ci, 'loc'):
                        ci_vals = ci.loc[varname].tolist()
                    else:
                        # fallback: try array indexing where varname appears in index
                        ci_vals = ci[varname]
                    entry['ci_lower'] = float(ci_vals[0])
                    entry['ci_upper'] = float(ci_vals[1])
                except Exception:
                    pass
            # n observations
            try:
                entry['n_obs'] = int(getattr(model, 'nobs', None))
            except Exception:
                entry['n_obs'] = None
        except Exception:
            # keep NaNs if any attribute access fails
            pass

        # record whether effect is statistically significant (two-sided alpha=0.05)
        is_signif = False
        if not np.isnan(entry['p']):
            is_signif = entry['p'] < 0.05
            sig_flags.append((spec_name, is_signif, entry['coef']))
        stats[spec_name] = entry

    # Build overall conclusion based on available specs
    conclusion = ""
    if not stats:
        conclusion = "No model results found in model_output."
    else:
        # Prefer linear model for direct interpretation; fall back to log if linear missing
        chosen_spec = None
        if 'model_linear' in stats:
            chosen_spec = stats['model_linear']
        else:
            chosen_spec = stats.get('model_log', None)

        if chosen_spec is None:
            conclusion = "Models present but unable to extract the student-teacher ratio coefficient."
        else:
            coef = chosen_spec['coef']
            p = chosen_spec['p']
            ci_lower = chosen_spec['ci_lower']
            ci_upper = chosen_spec['ci_upper']
            spec_label = chosen_spec['specification']

            if np.isnan(coef):
                conclusion = f"Could not extract coefficient for {chosen_spec['variable']} in {spec_label}."
            else:
                # Interpret sign: negative coef means higher ratio (more students per teacher) -> lower scores,
                # hence lower ratio (fewer students per teacher) associated with higher scores.
                direction = "negative" if coef < 0 else "positive"
                magnitude_text = ""
                if spec_label == 'model_linear':
                    magnitude_text = f"Point estimate: {coef:.3f} score points per one additional student-per-teacher."
                else:
                    magnitude_text = (f"Point estimate: {coef:.3f} change in AvgScore per 1-unit increase in log(student-teacher ratio). "
                                      "This approximates a percent-change interpretation on a log scale.")

                if np.isnan(p):
                    signif_text = "p-value not available, cannot assess statistical significance."
                else:
                    if p < 0.05:
                        signif_text = f"Statistically significant (p = {p:.3g})."
                    else:
                        signif_text = f"Not statistically significant (p = {p:.3g})."

                ci_text = ""
                if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                    ci_text = f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]."

                # Final statement tailored to the research question
                if (not np.isnan(p)) and (p < 0.05):
                    if coef < 0:
                        final_stmt = ("Evidence indicates that a lower student-teacher ratio (fewer students per teacher) "
                                      "is associated with higher average test scores.")
                    else:
                        final_stmt = ("Evidence indicates that a lower student-teacher ratio is associated with lower average test scores "
                                      "(coefficient has opposite sign).")
                else:
                    # no statistically significant evidence
                    if coef < 0:
                        final_stmt = ("The point estimate is negative (consistent with the hypothesis that lower student-teacher "
                                      "ratios are associated with higher scores), but the effect is not statistically significant. "
                                      "Therefore we do not have strong evidence to conclude an association.")
                    else:
                        final_stmt = ("The point estimate is positive (opposite the hypothesis) and not statistically significant; "
                                      "we do not have evidence that student-teacher ratio is associated with average scores in the expected direction.")

                conclusion = "Chosen specification: " + spec_label + ". " + magnitude_text + " " + ci_text + " " + signif_text + " " + final_stmt

    results = {
        'specification_stats': stats,
        'overall_conclusion': conclusion
    }

    description = (
        "Extracted coefficient, standard error, t-value, p-value, 95% confidence interval, and sample size "
        "for the student-teacher ratio from the linear and log specifications (when available). "
        "Interpretation: negative coefficient means that higher student-teacher ratio (more students per teacher) "
        "is associated with lower average scores (so a lower ratio would be associated with higher scores). "
        "The function reports whether the effect is statistically significant (two-sided alpha=0.05) and returns "
        "a concise conclusion based on the preferred linear specification when available."
    )

    return {"object": results, "description": description}