def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of instructor beauty (beauty_z and beauty_z_sq)
    from a statsmodels RegressionResultsWrapper fitted object.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Human-readable interpretation string"
      }
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Ensure we can access params, pvalues, conf_int, cov_params
    params = pd.Series(res.params)
    pvalues = pd.Series(res.pvalues)
    try:
        conf = res.conf_int()
        # conf may be ndarray; convert to DataFrame with same index as params
        if not isinstance(conf, pd.DataFrame):
            conf = pd.DataFrame(conf, index=params.index, columns=["2.5%", "97.5%"])
    except Exception:
        # fallback: make NaNs
        conf = pd.DataFrame(np.nan, index=params.index, columns=["2.5%", "97.5%"])

    # Covariance matrix for linear combinations
    cov = res.cov_params()
    cov_df = cov if isinstance(cov, pd.DataFrame) else pd.DataFrame(cov, index=params.index, columns=params.index)

    # Helper to check presence
    def has_param(name):
        return name in params.index

    # Names we need
    name_lin = 'beauty_z'
    name_quad = 'beauty_z_sq'

    if not (has_param(name_lin) and has_param(name_quad)):
        return {
            "object": None,
            "description": f"Model output does not contain both '{name_lin}' and '{name_quad}' parameters. Found params: {list(params.index)}"
        }

    # Extract coefficients and stats
    b1 = float(params[name_lin])
    b2 = float(params[name_quad])
    p1 = float(pvalues[name_lin])
    p2 = float(pvalues[name_quad])
    ci1 = tuple(conf.loc[name_lin].values) if name_lin in conf.index else (np.nan, np.nan)
    ci2 = tuple(conf.loc[name_quad].values) if name_quad in conf.index else (np.nan, np.nan)

    # Joint test that both beauty terms equal zero
    # Build restriction matrix R for [beta_beauty = 0; beta_beauty_sq = 0]
    k = len(params)
    R = np.zeros((2, k))
    param_list = list(params.index)
    R[0, param_list.index(name_lin)] = 1.0
    R[1, param_list.index(name_quad)] = 1.0
    try:
        ftest = res.f_test(R)
        joint_p = float(ftest.pvalue)
        joint_f = float(getattr(ftest, 'fvalue', np.nan))
    except Exception:
        # fallback: NaN
        joint_p = np.nan
        joint_f = np.nan

    # Marginal effects and standard errors for specific contrasts
    # 1) Marginal effect at x = 0 (derivative) is b1
    # 2) Marginal effect at x = 1 is b1 + 2*b2*1 = b1 + 2*b2
    # But often we want predicted change from 0 to +1: delta(0->1) = b1*(1) + b2*(1^2) = b1 + b2
    # Change from -1 to +1: f(1)-f(-1) = 2*b1 (quadratic cancels)
    # We'll compute SE for delta_0to1 = [0,...,1,1] dot params where vector selects b1 and b2.
    idx_b1 = param_list.index(name_lin)
    idx_b2 = param_list.index(name_quad)

    # Delta 0 -> 1
    R_delta01 = np.zeros(k)
    R_delta01[idx_b1] = 1.0  # coefficient on b1 times x (1)
    R_delta01[idx_b2] = 1.0  # coefficient on b2 times x^2 (1)
    delta01 = float(np.dot(R_delta01, params.values))
    var_delta01 = float(R_delta01.dot(cov_df.values).dot(R_delta01))
    se_delta01 = float(np.sqrt(var_delta01)) if var_delta01 >= 0 else float('nan')
    t_delta01 = delta01 / se_delta01 if se_delta01 and not np.isnan(se_delta01) else np.nan
    df_resid = getattr(res, 'df_resid', None)
    if df_resid is None or np.isnan(df_resid):
        p_delta01 = np.nan
    else:
        p_delta01 = float(2.0 * (1.0 - stats.t.cdf(abs(t_delta01), df=df_resid)))

    # Change from -1 to +1
    delta_m1_p1 = 2.0 * b1
    # SE for 2*b1 is 2 * se(b1) using var scaling
    se_b1 = float(np.sqrt(cov_df.values[idx_b1, idx_b1]))
    se_delta_m1_p1 = abs(2.0) * se_b1
    t_delta_m1_p1 = delta_m1_p1 / se_delta_m1_p1 if se_delta_m1_p1 and not np.isnan(se_delta_m1_p1) else np.nan
    if df_resid is None or np.isnan(df_resid):
        p_delta_m1_p1 = np.nan
    else:
        p_delta_m1_p1 = float(2.0 * (1.0 - stats.t.cdf(abs(t_delta_m1_p1), df=df_resid)))

    # Extremum of the quadratic: x* = -b1 / (2*b2) (if b2 != 0)
    if b2 != 0:
        x_star = -b1 / (2.0 * b2)
        # Predicted change from 0 to x_star: delta = b1*x_star + b2*x_star^2
        delta_xstar = b1 * x_star + b2 * (x_star ** 2)
        # SE for that linear combination: vector with entries for b1 and b2
        R_xstar = np.zeros(k)
        R_xstar[idx_b1] = x_star
        R_xstar[idx_b2] = x_star ** 2
        var_xstar = float(R_xstar.dot(cov_df.values).dot(R_xstar))
        se_xstar = float(np.sqrt(var_xstar)) if var_xstar >= 0 else float('nan')
        t_xstar = delta_xstar / se_xstar if se_xstar and not np.isnan(se_xstar) else np.nan
        if df_resid is None or np.isnan(df_resid):
            p_xstar = np.nan
        else:
            p_xstar = float(2.0 * (1.0 - stats.t.cdf(abs(t_xstar), df=df_resid)))
    else:
        x_star = np.nan
        delta_xstar = np.nan
        se_xstar = np.nan
        p_xstar = np.nan
        t_xstar = np.nan

    # Build return object dictionary with numeric results
    result_object = {
        "coef_beauty": b1,
        "se_beauty": float(np.sqrt(cov_df.values[idx_b1, idx_b1])),
        "p_beauty": p1,
        "ci_beauty_95": (float(ci1[0]), float(ci1[1])),
        "coef_beauty_sq": b2,
        "se_beauty_sq": float(np.sqrt(cov_df.values[idx_b2, idx_b2])),
        "p_beauty_sq": p2,
        "ci_beauty_sq_95": (float(ci2[0]), float(ci2[1])),
        "joint_f": joint_f,
        "joint_p": joint_p,
        "delta_0_to_1": delta01,
        "se_delta_0_to_1": se_delta01,
        "p_delta_0_to_1": p_delta01,
        "delta_minus1_to_plus1": delta_m1_p1,
        "se_delta_minus1_to_plus1": se_delta_m1_p1,
        "p_delta_minus1_to_plus1": p_delta_m1_p1,
        "extremum_x": x_star,
        "change_at_extremum_vs_0": delta_xstar,
        "se_change_at_extremum": se_xstar,
        "p_change_at_extremum": p_xstar,
        "params_index": param_list
    }

    # Human-readable description
    def fmt(x):
        try:
            return f"{x:.3f}"
        except Exception:
            return str(x)

    desc_lines = []
    desc_lines.append("Effect of beauty on teaching evaluations (dependent variable: eval, scale 1-5).")
    desc_lines.append(f"Linear term (beauty_z): coef = {fmt(b1)}, SE = {fmt(result_object['se_beauty'])}, p = {fmt(p1)}; 95% CI = ({fmt(result_object['ci_beauty_95'][0])}, {fmt(result_object['ci_beauty_95'][1])}).")
    desc_lines.append(f"Quadratic term (beauty_z_sq): coef = {fmt(b2)}, SE = {fmt(result_object['se_beauty_sq'])}, p = {fmt(p2)}; 95% CI = ({fmt(result_object['ci_beauty_sq_95'][0])}, {fmt(result_object['ci_beauty_sq_95'][1])}).")
    if not np.isnan(joint_p):
        desc_lines.append(f"Joint test that both beauty coefficients = 0: F = {fmt(joint_f)}, p = {fmt(joint_p)}.")
    else:
        desc_lines.append("Could not compute joint test p-value for the two beauty terms.")

    desc_lines.append(f"Predicted change in eval from 0 to +1 SD in beauty: {fmt(delta01)} (SE = {fmt(se_delta01)}, p = {fmt(p_delta01)}).")
    desc_lines.append(f"Predicted change in eval from -1 SD to +1 SD in beauty: {fmt(delta_m1_p1)} (SE = {fmt(se_delta_m1_p1)}, p = {fmt(p_delta_m1_p1)}).")

    if not np.isnan(x_star):
        desc_lines.append(f"The quadratic's extremum is at beauty_z = {fmt(x_star)}. Predicted change from 0 to that extremum = {fmt(delta_xstar)} (SE = {fmt(se_xstar)}, p = {fmt(p_xstar)}).")
        # Note whether extremum is within typical +/-3 SD range
        within_range = (-3 <= x_star <= 3)
        desc_lines.append(f"Extremum within +/-3 SD range: {within_range}.")
    else:
        desc_lines.append("No quadratic curvature detected (beauty_z_sq = 0), so no extremum computed.")

    # Summarize statistical conclusion in plain language
    alpha = 0.05
    sig_individual = (p1 < alpha) or (p2 < alpha)
    sig_joint = (not np.isnan(joint_p)) and (joint_p < alpha)
    if sig_joint:
        concl = "There is evidence that beauty (considering both linear and quadratic terms jointly) is associated with teaching evaluations (joint test p < 0.05)."
    elif sig_individual:
        concl = "At least one beauty term is statistically significant at p < 0.05, indicating some relationship; check individual coefficients above."
    else:
        concl = "No statistically significant evidence that beauty (either linear or quadratic term) is associated with teaching evaluations at the 0.05 level."

    desc_lines.append(concl)

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}