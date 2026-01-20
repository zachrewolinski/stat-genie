def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, confidence intervals, and significance
    for the main variable of interest (beauty_z), its interaction with gender
    (beauty_gender_interaction), and the implied marginal effects for males and females.

    Returns:
      {
        "object": { ... },           # numeric results for programmatic use
        "description": "..."         # brief plain-language interpretation
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output  # statsmodels RegressionResultsWrapper

    # Names expected in the model
    name_beauty = 'beauty_z'
    name_inter = 'beauty_gender_interaction'

    params = res.params
    cov = res.cov_params()  # uses the fitted model's covariance (cluster-robust if so fitted)

    # Helper to extract single-parameter results (uses model's reported pvalue if available)
    def single_param_stats(name):
        if name not in params.index:
            return None
        est = float(params.loc[name])
        # Standard error from covariance matrix (sqrt of variance on diagonal)
        se = float(np.sqrt(cov.loc[name, name]))
        # Use z-statistic (large-sample approx); also provide model's own t/p if present
        z = est / se if se != 0 else np.nan
        p_z = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
        # 95% CI using normal approx
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se
        # Try to include model's reported t and p if available
        t_value = float(res.tvalues[name]) if name in res.tvalues.index else None
        p_value_model = float(res.pvalues[name]) if name in res.pvalues.index else None

        return {
            "name": name,
            "estimate": est,
            "se": se,
            "z_or_t": z if t_value is None else t_value,
            "p_z": p_z,
            "p_modelreport": p_value_model,
            "ci_95": [ci_low, ci_high]
        }

    beauty_stats = single_param_stats(name_beauty)
    inter_stats = single_param_stats(name_inter)

    # Compute marginal effects:
    # male (gender_F=0): effect = coef(beauty_z)
    # female (gender_F=1): effect = coef(beauty_z) + coef(beauty_gender_interaction)
    def linear_combination_stats(coef_names, weights):
        # coef_names: list of parameter names
        # weights: list/array of weights same length
        w = np.asarray(weights, dtype=float)
        # check names exist
        for n in coef_names:
            if n not in params.index:
                return None
        # build vector of coefficients
        ests = np.array([params.loc[n] for n in coef_names], dtype=float)
        est = float(np.dot(w, ests))
        # build covariance submatrix and compute variance of linear combo
        cov_sub = cov.loc[coef_names, coef_names].values
        var = float(w @ cov_sub @ w)
        se = float(np.sqrt(var)) if var >= 0 else np.nan
        z = est / se if se != 0 else np.nan
        p_z = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se
        return {
            "estimate": est,
            "se": se,
            "z": z,
            "p_z": p_z,
            "ci_95": [ci_low, ci_high],
            "components": dict(zip(coef_names, [float(params.loc[n]) for n in coef_names]))
        }

    male_effect = linear_combination_stats([name_beauty], [1.0])
    female_effect = linear_combination_stats([name_beauty, name_inter], [1.0, 1.0])

    # Build object to return
    result_object = {
        "beauty_z": beauty_stats,
        "beauty_gender_interaction": inter_stats,
        "marginal_effect_male": male_effect,
        "marginal_effect_female": female_effect
    }

    # Interpretation / description (concise)
    def sig_label(p):
        if p is None:
            return "n/a"
        if p < 0.001:
            return "*** (p < 0.001)"
        if p < 0.01:
            return "** (p < 0.01)"
        if p < 0.05:
            return "* (p < 0.05)"
        return "not significant (p >= 0.05)"

    # Determine significance messages
    beauty_sig = sig_label(beauty_stats["p_modelreport"] if beauty_stats else None)
    inter_sig = sig_label(inter_stats["p_modelreport"] if inter_stats else None)
    male_sig = sig_label(male_effect["p_z"] if male_effect else None)
    female_sig = sig_label(female_effect["p_z"] if female_effect else None)

    desc_lines = []
    if beauty_stats:
        desc_lines.append(
            f"Main effect (beauty_z): estimate = {beauty_stats['estimate']:.3f}, "
            f"SE = {beauty_stats['se']:.3f}, p (model) = {beauty_stats.get('p_modelreport'):.3g} -> {beauty_sig}."
        )
    else:
        desc_lines.append("Variable 'beauty_z' not found in model results.")

    if inter_stats:
        desc_lines.append(
            f"Interaction (beauty_gender_interaction): estimate = {inter_stats['estimate']:.3f}, "
            f"SE = {inter_stats['se']:.3f}, p (model) = {inter_stats.get('p_modelreport'):.3g} -> {inter_sig}."
        )
    else:
        desc_lines.append("Interaction term 'beauty_gender_interaction' not found in model results.")

    if male_effect:
        desc_lines.append(
            f"Marginal effect for males (gender_F=0): estimate = {male_effect['estimate']:.3f}, "
            f"SE = {male_effect['se']:.3f}, p (z) = {male_effect['p_z']:.3g} -> {male_sig}."
        )
    if female_effect:
        desc_lines.append(
            f"Marginal effect for females (gender_F=1): estimate = {female_effect['estimate']:.3f}, "
            f"SE = {female_effect['se']:.3f}, p (z) = {female_effect['p_z']:.3g} -> {female_sig}."
        )

    # Final short conclusion about whether beauty matters
    # Use p<0.05 for decision
    conclusion = "Conclusion: "
    if (male_effect and male_effect["p_z"] < 0.05) or (female_effect and female_effect["p_z"] < 0.05):
        conclusion += "There is evidence that instructor physical attractiveness (beauty) is associated with teaching evaluations for at least one gender."
        if male_effect and male_effect["p_z"] < 0.05 and female_effect and female_effect["p_z"] < 0.05:
            conclusion += " Effects are statistically significant for both males and females."
        elif male_effect and male_effect["p_z"] < 0.05:
            conclusion += " Effect is statistically significant for males but not females."
        else:
            conclusion += " Effect is statistically significant for females but not males."
    else:
        conclusion += "No statistically significant association between instructor beauty and evaluations was detected for either gender (at alpha = 0.05)."

    description = "\n".join(desc_lines + ["", conclusion])

    return {"object": result_object, "description": description}