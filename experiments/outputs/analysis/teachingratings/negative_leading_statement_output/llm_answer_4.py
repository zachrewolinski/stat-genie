def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of instructor attractiveness (beauty_z)
    on course evaluations from the provided model_output dictionary.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Concise interpretation string"
      }
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    ols_res = model_output.get('ols_cluster')
    fe_res = model_output.get('fe_ols')

    if ols_res is None:
        raise ValueError("model_output does not contain 'ols_cluster' results.")

    # Safely extract coefficient and se (prefer values from the fitted object)
    param_name = 'beauty_z'
    params = getattr(ols_res, 'params', None)
    bse = getattr(ols_res, 'bse', None)
    tvals = getattr(ols_res, 'tvalues', None)
    pvals = getattr(ols_res, 'pvalues', None)

    coef = float(params.get(param_name)) if (params is not None and param_name in params.index) else float(model_output.get('beauty_coef_ols_cluster', np.nan))
    se = float(bse.get(param_name)) if (bse is not None and param_name in bse.index) else float(model_output.get('beauty_se_ols_cluster', np.nan))
    t_value = float(tvals.get(param_name)) if (tvals is not None and param_name in tvals.index) else (coef / se if se and not np.isnan(se) else np.nan)
    p_value = float(pvals.get(param_name)) if (pvals is not None and param_name in pvals.index) else np.nan

    # 95% confidence interval
    try:
        ci_all = ols_res.conf_int()
        if hasattr(ci_all, 'loc'):
            ci_lower, ci_upper = ci_all.loc[param_name].tolist()
        else:
            # ci_all is ndarray; find index
            idx = list(ols_res.params.index).index(param_name)
            ci_lower, ci_upper = ci_all[idx].tolist()
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
    except Exception:
        ci_lower = ci_upper = np.nan

    # Sample size and number of professors (if provided)
    n_obs = int(model_output.get('n_obs', getattr(ols_res, 'nobs', np.nan)))
    n_professors = int(model_output.get('n_professors', np.nan)) if model_output.get('n_professors', None) is not None else (int(len(np.unique(getattr(ols_res.model.data, 'row_labels', [])))) if hasattr(ols_res.model, 'data') else None)

    # Standardized effect relative to outcome SD (compute SD of eval from model's endog if available)
    try:
        endog = getattr(ols_res.model, 'endog', None)
        if endog is not None:
            sd_eval = float(np.std(endog, ddof=1))
            standardized_effect = coef / sd_eval if sd_eval != 0 else np.nan
        else:
            sd_eval = standardized_effect = np.nan
    except Exception:
        sd_eval = standardized_effect = np.nan

    # Percent of the 1-5 evaluation scale covered by the coefficient
    percent_of_scale = (coef / 4.0) * 100.0 if coef is not None and not np.isnan(coef) else np.nan

    # Fixed-effects model estimate (robustness check)
    fe_coef = fe_p = fe_se = None
    if fe_res is not None:
        fe_params = getattr(fe_res, 'params', None)
        fe_bse = getattr(fe_res, 'bse', None)
        fe_pvals = getattr(fe_res, 'pvalues', None)
        if fe_params is not None and param_name in fe_params.index:
            fe_coef = float(fe_params.get(param_name))
            fe_se = float(fe_bse.get(param_name)) if (fe_bse is not None and param_name in fe_bse.index) else None
            fe_p = float(fe_pvals.get(param_name)) if (fe_pvals is not None and param_name in fe_pvals.index) else None

    result_object = {
        "coef_ols_cluster": coef,
        "se_ols_cluster": se,
        "t_value_ols_cluster": t_value,
        "p_value_ols_cluster": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_obs": n_obs,
        "n_professors": n_professors,
        "sd_eval": sd_eval,
        "standardized_effect (coef / sd_eval)": standardized_effect,
        "percent_of_1-5_scale": percent_of_scale,
        "fe_coef": fe_coef,
        "fe_se": fe_se,
        "fe_p_value": fe_p
    }

    # Build a concise human-readable interpretation
    desc = (
        f"OLS (clustered by professor) estimate: a one standard-deviation increase in instructor attractiveness "
        f"(beauty_z) is associated with an increase of {coef:.3f} points in the overall course evaluation (scale 1-5), "
        f"SE = {se:.3f}, t = {t_value:.2f}, p = {p_value:.3g}. "
        f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]. "
        f"Sample: {n_obs} course observations from {n_professors} professors. "
    )

    # Add interpretation of magnitude
    if not np.isnan(standardized_effect):
        desc += (
            f"This effect corresponds to {standardized_effect:.3f} SDs of the evaluation score, "
            f"or {percent_of_scale:.2f}% of the full 1-5 scale range. "
        )
    else:
        desc += f"(Could not compute standardized effect because eval SD was unavailable.) "

    # Note on robustness check
    if fe_coef is not None:
        desc += (
            f"Fixed-effects (within-professor) estimate: coef = {fe_coef:.3f}"
            + (f", SE = {fe_se:.3f}, p = {fe_p:.3g}." if fe_se is not None else ".")
            + " This checks whether the relationship holds using within-instructor variation."
        )

    return {"object": result_object, "description": desc}