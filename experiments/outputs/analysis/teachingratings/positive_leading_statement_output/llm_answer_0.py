def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of instructor beauty (beauty_z) on teaching evaluations
    from the provided statsmodels RegressionResultsWrapper objects.

    Parameters
    ----------
    model_output : dict
        Expected keys: 'model_simple', 'model_controls_clustered', 'model_prof_fe'
        Each value should be a statsmodels RegressionResultsWrapper (or None).

    Returns
    -------
    dict with keys:
      - "object": dict mapping model name -> extracted numeric statistics about 'beauty_z'
      - "description": a human-readable summary of what the extracted numbers mean
    """
    import numpy as np

    out = {}
    summary_lines = []

    for key in ['model_simple', 'model_controls_clustered', 'model_prof_fe']:
        res = model_output.get(key, None)
        if res is None:
            out[key] = None
            summary_lines.append(f"{key}: model not provided / was None.")
            continue

        try:
            params = res.params
        except Exception as e:
            out[key] = {"error": f"Could not read params: {e}"}
            summary_lines.append(f"{key}: could not read params ({e}).")
            continue

        if 'beauty_z' not in params.index:
            out[key] = {"error": "No coefficient named 'beauty_z' in model."}
            summary_lines.append(f"{key}: model does not contain a 'beauty_z' coefficient.")
            continue

        coef = float(params['beauty_z'])
        # standard error, p-value, and t-value
        try:
            se = float(res.bse['beauty_z'])
        except Exception:
            se = None
        try:
            pval = float(res.pvalues['beauty_z'])
        except Exception:
            pval = None
        try:
            tval = float(res.tvalues['beauty_z'])
        except Exception:
            tval = None

        # 95% CI (approximate using normal 1.96*se if se available)
        if se is not None:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            ci_lower = ci_upper = None

        # number of observations and R-squared if available
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None
        try:
            rsq = float(res.rsquared)
        except Exception:
            rsq = None

        # effect relative to outcome SD (how many SDs change in eval per 1 SD beauty)
        try:
            y = np.asarray(res.model.endog)
            y_sd = float(np.nanstd(y, ddof=1))
            effect_in_sd = coef / y_sd if y_sd != 0 else None
        except Exception:
            y_sd = None
            effect_in_sd = None

        out[key] = {
            "coef": coef,
            "std_err": se,
            "t_value": tval,
            "p_value": pval,
            "95ci_lower": ci_lower,
            "95ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": rsq,
            "eval_sd": y_sd,
            "effect_in_eval_sd_per_beauty_sd": effect_in_sd,
        }

        # Build a summary line for this model
        sig_text = "statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant"
        ci_text = f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]" if (ci_lower is not None and ci_upper is not None) else "CI unavailable"
        effect_sd_text = (f"{effect_in_sd:.3f} SDs change in eval per 1 SD change in beauty"
                          if effect_in_sd is not None else "effect in SD-units unavailable")

        summary_lines.append(
            f"{key}: coef={coef:.4f}, se={se:.4f} (t={tval:.2f}, p={pval:.3f}) -> {sig_text}; "
            f"{ci_text}; n={nobs}; r2={rsq if rsq is not None else 'NA'}; {effect_sd_text}."
        )

    description = (
        "The returned 'object' is a dict keyed by model name (model_simple, model_controls_clustered, "
        "model_prof_fe). For each model it reports:\n"
        "- coef: estimated change in course evaluation (eval, 1-5 scale) associated with a 1 SD increase in beauty_z\n"
        "- std_err, t_value, p_value: inferential statistics for the beauty_z coefficient\n"
        "- 95ci_lower / 95ci_upper: approximate 95% confidence interval (coef ± 1.96*se)\n"
        "- n_obs: number of observations used to fit that model\n"
        "- r_squared: model R-squared (where available)\n"
        "- eval_sd: empirical SD of the eval outcome in the model sample\n"
        "- effect_in_eval_sd_per_beauty_sd: coef divided by eval_sd (how many SDs of eval change per 1 SD beauty)\n\n"
        "Interpretation guidance: the 'coef' is directly interpretable because beauty_z is standardized: "
        "it gives the expected change in the course evaluation score (on the 1-5 scale) for a one-standard-deviation "
        "increase in instructor beauty, holding controls (if present) constant. Use the p_value and 95% CI to judge "
        "statistical significance and precision. The effect_in_eval_sd_per_beauty_sd shows the effect size in standard-"
        "deviation units of the outcome (useful for comparing across outcomes).\n\n"
        "Quick per-model summary:\n" + "\n".join(summary_lines)
    )

    return {"object": out, "description": description}