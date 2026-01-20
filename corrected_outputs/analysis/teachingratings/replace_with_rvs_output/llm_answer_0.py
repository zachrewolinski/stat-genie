def extract_final_answer(model_output):
    """
    Extract key statistics for the 'beauty_std' coefficient from a statsmodels
    RegressionResultsWrapper (cluster-robust OLS) and return a concise
    result + interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, n_obs, n_clusters, significance)
      - "description": human-readable explanation of what the numbers mean
    """
    var = 'beauty_std'
    res = model_output

    # Helper to extract confidence interval robustly
    try:
        conf = res.conf_int()
        if var in conf.index:
            ci_lower, ci_upper = float(conf.loc[var, 0]), float(conf.loc[var, 1])
        else:
            # fallback: try to find row by name partially or by position
            try:
                # if index is an array of names, find matching
                idx = [i for i, name in enumerate(conf.index) if name == var]
                if idx:
                    ci_lower, ci_upper = float(conf.iloc[idx[0], 0]), float(conf.iloc[idx[0], 1])
                else:
                    raise KeyError
            except Exception:
                # as last resort, take the row corresponding to the parameter position
                raise KeyError(f"Could not find CI row for variable '{var}'")
    except Exception as e:
        raise ValueError(f"Failed to compute confidence interval for '{var}': {e}")

    # Extract coefficient, SE, t, p
    try:
        coef = float(res.params[var])
        se = float(res.bse[var])
        tval = float(res.tvalues[var])
        pval = float(res.pvalues[var])
    except Exception as e:
        raise ValueError(f"Failed to extract coefficient/stats for '{var}': {e}")

    # Sample size
    try:
        n_obs = int(res.nobs)
    except Exception:
        n_obs = None

    # Number of clusters (unique prof ids) if available from the model's data frame
    n_clusters = None
    try:
        df = res.model.data.frame
        if 'prof' in df.columns:
            n_clusters = int(df['prof'].nunique())
    except Exception:
        # leave as None if not available
        n_clusters = None

    # Statistical significance at conventional alpha = 0.05
    significant_0_05 = (pval < 0.05)

    # Interpretation: beauty_std is standardized (1 SD). Coefficient = change in eval (1-5) per 1 SD increase.
    interp = (
        f"Estimated effect of beauty (per 1 SD increase) on course evaluation = {coef:.4f} points "
        f"(SE = {se:.4f}, t = {tval:.2f}, p = {pval:.3f}). 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"{'Statistically significant at p<0.05.' if significant_0_05 else 'Not statistically significant at conventional levels (p>=0.05).'}"
    )
    if n_obs is not None or n_clusters is not None:
        interp += f" Sample size (observations) = {n_obs if n_obs is not None else 'unknown'}; clusters (professors) = {n_clusters if n_clusters is not None else 'unknown'}."

    result_object = {
        'variable': var,
        'coefficient': coef,
        'std_error': se,
        't_value': tval,
        'p_value': pval,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'n_observations': n_obs,
        'n_clusters': n_clusters,
        'significant_0.05': significant_0_05
    }

    return {'object': result_object, 'description': interp}