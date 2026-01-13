def extract_final_answer(model_output):
    """
    Extract site-specific age slopes (log-odds change per year) and their statistics
    from a fitted statsmodels GLM (logistic) that included age_c * C(site_id).
    
    Returns:
      dict with keys:
        - "object": {
            "site_slopes": [ {site, slope_logodds, se, z, p, ci_lower, ci_upper,
                              odds_ratio, or_ci_lower, or_ci_upper}, ... ],
            "interaction_test": { "lr_chi2", "df_diff", "p_value" }  # likelihood-ratio test
          }
        - "description": human-readable explanation of what the numbers mean.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm, chi2
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    res = model_output  # GLMResultsWrapper expected

    # Try to retrieve the original dataframe used to fit the model
    try:
        data_df = res.model.data.frame.copy()
    except Exception:
        # fallback: try attribute data.orig_endog / exog won't give site labels; raise error
        raise RuntimeError("Could not retrieve the original DataFrame from the model object. "
                           "The function needs access to the data to reconstruct site levels "
                           "and to refit the reduced model for the interaction test.")

    # Determine site levels in their categorical order
    if 'site_id' not in data_df.columns:
        raise ValueError("Original data frame does not contain 'site_id' column.")
    site_levels = pd.Categorical(data_df['site_id']).categories.tolist()
    if len(site_levels) == 0:
        raise ValueError("No site levels found in 'site_id' column.")

    # Parameters and covariance matrix
    params = res.params
    cov = res.cov_params()

    # Ensure base age parameter exists
    if 'age_c' not in params.index:
        raise ValueError("Fitted model does not contain a main 'age_c' coefficient. "
                         "Ensure the model formula included 'age_c'.")

    base_name = 'age_c'
    base_coef = float(params[base_name])
    # prepare to find interaction parameter names
    # Patsy typically names interactions like 'age_c:C(site_id)[T.<level>]'
    param_index = list(params.index)

    # Map for site -> interaction param name (if exists)
    interaction_map = {}
    for pname in param_index:
        if pname.startswith('age_c:') and 'C(site_id)' in pname:
            # e.g., 'age_c:C(site_id)[T.X]'
            # Extract level portion
            interaction_map[pname] = pname

    # Alternatively, detect parameter names of form 'age_c:C(site_id)[T.<level>]'
    # We'll construct per-site slopes:
    site_results = []
    for i, site in enumerate(site_levels):
        if i == 0:
            # reference level: slope is base_coef
            slope = base_coef
            # variance is var(age_c)
            var = cov.loc[base_name, base_name]
        else:
            # interaction parameter name expected:
            inter_name = f'age_c:C(site_id)[T.{site}]'
            if inter_name in params.index:
                inter_coef = float(params[inter_name])
                slope = base_coef + inter_coef
                # Var(slope) = Var(base) + Var(inter) + 2*Cov(base,inter)
                var = (cov.loc[base_name, base_name]
                       + cov.loc[inter_name, inter_name]
                       + 2 * cov.loc[base_name, inter_name])
            else:
                # If exact interaction name not found, try alternative naming patterns
                # Try 'age_c:C(site_id)[T.%s]' already tried; try 'age_c:C(site_id)[%s]'
                inter_name_alt = f'age_c:C(site_id)[{site}]'
                if inter_name_alt in params.index:
                    inter_coef = float(params[inter_name_alt])
                    slope = base_coef + inter_coef
                    var = (cov.loc[base_name, base_name]
                           + cov.loc[inter_name_alt, inter_name_alt]
                           + 2 * cov.loc[base_name, inter_name_alt])
                else:
                    # No explicit interaction term found for this site: assume slope == base_coef
                    slope = base_coef
                    var = cov.loc[base_name, base_name]

        se = np.sqrt(float(var)) if var >= 0 else np.nan
        z = slope / se if se and not np.isnan(se) else np.nan
        p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_lower = slope - 1.96 * se if not np.isnan(se) else np.nan
        ci_upper = slope + 1.96 * se if not np.isnan(se) else np.nan
        odds_ratio = float(np.exp(slope)) if not np.isnan(slope) else np.nan
        or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan
        or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan

        site_results.append({
            "site": site,
            "slope_logodds_per_year": float(slope),
            "se": float(se),
            "z": float(z) if not np.isnan(z) else None,
            "p_value": float(p) if not np.isnan(p) else None,
            "ci_lower_logodds": float(ci_lower) if not np.isnan(ci_lower) else None,
            "ci_upper_logodds": float(ci_upper) if not np.isnan(ci_upper) else None,
            "odds_ratio_per_year": float(odds_ratio),
            "or_ci_lower": float(or_ci_lower),
            "or_ci_upper": float(or_ci_upper),
        })

    # Perform a likelihood-ratio test comparing full model (with interactions) to reduced model (without interactions).
    # Refit reduced model: remove the interaction age_c * C(site_id) -> keep additive age_c + C(site_id)
    formula_reduced = 'is_majority_choice ~ age_c + C(site_id) + is_boy + demo_order_majority_first'
    try:
        reduced = smf.glm(formula=formula_reduced, data=data_df, family=sm.families.Binomial()).fit()
        llf_full = float(res.llf)
        llf_reduced = float(reduced.llf)
        lr_chi2 = 2.0 * (llf_full - llf_reduced)
        # df difference equals number of additional parameters in full model relative to reduced
        df_diff = int(res.df_model) - int(reduced.df_model)
        if df_diff <= 0:
            p_value_lr = None
        else:
            p_value_lr = float(1 - chi2.cdf(lr_chi2, df_diff))
    except Exception as e:
        lr_chi2 = None
        df_diff = None
        p_value_lr = None

    output = {
        "site_slopes": site_results,
        "interaction_test": {
            "lr_chi2": float(lr_chi2) if lr_chi2 is not None else None,
            "df_diff": int(df_diff) if df_diff is not None else None,
            "p_value": float(p_value_lr) if p_value_lr is not None else None,
            "note": "Likelihood-ratio test compares full model (age_c * site) vs reduced (additive age_c + site). "
                    "A small p-value indicates that age-related change differs across sites (i.e., interaction present)."
        }
    }

    description = (
        "For each study site, 'slope_logodds_per_year' is the estimated change in log-odds of choosing the majority option "
        "for a one-year increase in age (age_c is mean-centered). 'se', 'z', and 'p_value' are the standard error, "
        "Wald z-statistic, and two-sided p-value testing whether the slope differs from zero. "
        "'ci_lower_logodds'/'ci_upper_logodds' are the 95% confidence interval on the log-odds scale; "
        "'odds_ratio_per_year' and its CI show the multiplicative change in odds per year. "
        "The interaction_test reports a likelihood-ratio test (chi-square, df, p) comparing the model with site-specific age slopes "
        "to a model with a single (shared) age slope; a small p-value suggests developmental trajectories differ across cultural contexts."
    )

    return {"object": output, "description": description}