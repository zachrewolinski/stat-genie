def extract_final_answer(model_output):
    """
    Extracts the estimated effects of instructor beauty on teaching evaluations
    from a fitted statsmodels OLS results object that used the formula:
      Eval ~ Beauty_c + Female + ... + Beauty_c:Female + C(InstructorID)
    and was fit with robust SEs (HC3).

    Returns a dict with:
      - "object": dict with numeric results for 'male', 'female', and 'interaction'
      - "description": textual interpretation of the extracted statistics
    """
    import numpy as np
    from scipy import stats

    res = model_output  # RegressionResultsWrapper

    params = res.params
    bse = res.bse
    pvals = res.pvalues
    # robust cov matrix (should reflect HC3 used in fit)
    cov = res.cov_params()

    # helper to get confint for a param name
    try:
        conf = res.conf_int(alpha=0.05)
        # conf may be a DataFrame or ndarray
        if hasattr(conf, "loc"):
            def _get_conf(name):
                return conf.loc[name].values.astype(float)
        else:
            def _get_conf(name):
                idx = params.index.get_loc(name)
                return conf[idx].astype(float)
    except Exception:
        # fallback: compute from coef +/- t_crit * se using df_resid
        def _get_conf(name):
            coef = params[name]
            se = bse[name]
            df = res.df_resid
            tcrit = stats.t.ppf(0.975, df) if df > 0 else 1.96
            return np.array([coef - tcrit * se, coef + tcrit * se])

    # Required parameter names
    name_beauty = 'Beauty_c'
    name_inter = 'Beauty_c:Female'  # name used by statsmodels for interaction term
    # If statsmodels used a slightly different naming for interaction (rare), try alternative
    if name_inter not in params.index and 'Female:Beauty_c' in params.index:
        name_inter = 'Female:Beauty_c'

    # Extract male effect (reference group Female=0)
    if name_beauty not in params.index:
        raise KeyError(f"Model does not contain expected parameter '{name_beauty}'. Available params: {list(params.index)}")
    coef_male = float(params[name_beauty])
    se_male = float(bse[name_beauty])
    t_male = float(coef_male / se_male) if se_male != 0 else np.nan
    # p-value from results (already computed)
    p_male = float(pvals.get(name_beauty, np.nan))
    ci_male = _get_conf(name_beauty)

    # Extract interaction term (difference in slope for females vs males)
    if name_inter in params.index:
        coef_inter = float(params[name_inter])
        se_inter = float(bse[name_inter])
        t_inter = float(coef_inter / se_inter) if se_inter != 0 else np.nan
        p_inter = float(pvals.get(name_inter, np.nan))
        ci_inter = _get_conf(name_inter)
    else:
        coef_inter = 0.0
        se_inter = np.nan
        t_inter = np.nan
        p_inter = np.nan
        ci_inter = np.array([np.nan, np.nan])

    # Compute female effect = Beauty_c + (Beauty_c:Female)
    coef_female = coef_male + coef_inter
    # variance and se for linear combination
    try:
        var_beauty = cov.loc[name_beauty, name_beauty]
        if name_inter in cov.index:
            cov_beauty_inter = cov.loc[name_beauty, name_inter]
            var_inter = cov.loc[name_inter, name_inter]
            var_female = var_beauty + var_inter + 2 * cov_beauty_inter
        else:
            var_female = var_beauty  # no interaction term present
        se_female = float(np.sqrt(var_female)) if var_female >= 0 else float(np.nan)
    except Exception:
        # fallback to NaNs if cov matrix indexing fails
        se_female = float(np.nan)

    t_female = float(coef_female / se_female) if (not np.isnan(se_female) and se_female != 0) else np.nan
    # p-value using t-distribution with df_resid
    df = res.df_resid if hasattr(res, "df_resid") else None
    if df is None or df <= 0 or np.isnan(t_female):
        p_female = np.nan
    else:
        p_female = float(2 * stats.t.sf(abs(t_female), df))

    # Confidence interval for female effect
    if not np.isnan(se_female) and df is not None and df > 0:
        tcrit = stats.t.ppf(0.975, df)
        ci_female = np.array([coef_female - tcrit * se_female, coef_female + tcrit * se_female])
    else:
        ci_female = np.array([np.nan, np.nan])

    # Build output object
    output_obj = {
        "male": {
            "coef": coef_male,
            "se": se_male,
            "t": t_male,
            "p": p_male,
            "95%_ci": [float(ci_male[0]), float(ci_male[1])]
        },
        "female": {
            "coef": coef_female,
            "se": se_female,
            "t": t_female,
            "p": p_female,
            "95%_ci": [float(ci_female[0]) if not np.isnan(ci_female[0]) else None,
                       float(ci_female[1]) if not np.isnan(ci_female[1]) else None]
        },
        "interaction (Beauty_c:Female)": {
            "coef": coef_inter,
            "se": se_inter,
            "t": t_inter,
            "p": p_inter,
            "95%_ci": [float(ci_inter[0]), float(ci_inter[1])]
        },
        "notes": "Effects are in units of course evaluation points (scale 1-5) per one-unit increase in mean-centered beauty score."
    }

    # Short textual interpretation
    desc_lines = []
    desc_lines.append(f"Estimated effect of beauty for male instructors (reference group): coef = {coef_male:.4f}, SE = {se_male:.4f}, t = {t_male:.3f}, p = {p_male:.3g}, 95% CI = [{ci_male[0]:.4f}, {ci_male[1]:.4f}].")
    if not np.isnan(coef_inter):
        desc_lines.append(f"Interaction (Beauty x Female): coef = {coef_inter:.4f}, SE = {se_inter:.4f}, t = {t_inter:.3f}, p = {p_inter:.3g}, 95% CI = [{ci_inter[0]:.4f}, {ci_inter[1]:.4f}].")
        desc_lines.append(f"Estimated effect of beauty for female instructors (sum of beauty + interaction): coef = {coef_female:.4f}, SE ≈ {se_female:.4f}, t ≈ {t_female:.3f}, p ≈ {p_female:.3g}, 95% CI ≈ [{ci_female[0]:.4f}, {ci_female[1]:.4f}].")
    else:
        desc_lines.append("No interaction term present; effect for females equals effect for males.")

    desc_lines.append("Interpretation: these coefficients are change in overall course evaluation (1-5) for a one-unit change in mean-centered beauty. Statistical significance (p < 0.05) indicates evidence that beauty predicts evaluations for the given group; a significant interaction indicates the beauty effect differs by gender.")

    description = " ".join(desc_lines)

    return {"object": output_obj, "description": description}