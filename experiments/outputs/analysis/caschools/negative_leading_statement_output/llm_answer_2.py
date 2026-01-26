def extract_final_answer(model_output):
    """
    Extracts the marginal effect of StudentTeacherRatio on AcademicScore from a fitted
    statsmodels RegressionResultsWrapper that included StudentTeacherRatio and its square
    (I(StudentTeacherRatio**2)) among the regressors.

    Returns a dictionary with:
      - "object": a dict containing coefficients, p-values, the marginal effect at the
                  sample mean of StudentTeacherRatio, its SE, t-stat, p-value and 95% CI,
                  plus the sample mean of StudentTeacherRatio.
      - "description": a short interpretation answering whether a lower student-teacher
                       ratio is associated with higher academic performance (based on
                       the sign and statistical significance of the marginal effect).
    """
    import numpy as np

    res = model_output  # expected to be a statsmodels RegressionResultsWrapper

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels result (missing .params)")

    # Get parameter names
    try:
        param_names = list(res.model.exog_names)
    except Exception:
        param_names = list(res.params.index)

    # Identify the linear and quadratic term names/indices robustly
    lin_idx = None
    quad_idx = None
    for i, nm in enumerate(param_names):
        # exact match for linear term
        if nm == "StudentTeacherRatio":
            lin_idx = i
        # detect the I(...**2) quadratic term (various spacing possible)
        if "StudentTeacherRatio" in nm and ("I(" in nm or "**" in nm or "StudentTeacherRatio" in nm and nm != "StudentTeacherRatio"):
            # prefer terms that include a power indicator
            if "I(" in nm or "**" in nm or "StudentTeacherRatio**2" in nm or "StudentTeacherRatio ** 2" in nm:
                quad_idx = i

    # Fallbacks if exact names not found: pick any parameters containing the variable
    if lin_idx is None:
        for i, nm in enumerate(param_names):
            if "StudentTeacherRatio" in nm and (("**" not in nm) and ("I(" not in nm)):
                lin_idx = i
                break
    if quad_idx is None:
        # pick another param that contains StudentTeacherRatio but is not the linear one
        for i, nm in enumerate(param_names):
            if "StudentTeacherRatio" in nm and i != lin_idx:
                quad_idx = i
                break

    if lin_idx is None or quad_idx is None:
        raise ValueError("Could not find both linear and quadratic StudentTeacherRatio terms in the model's parameters. "
                         "Found param names: " + ", ".join(param_names))

    # Compute sample mean of StudentTeacherRatio from model.exog if available, otherwise try to recover from data frame
    try:
        exog = np.asarray(res.model.exog)
        r_mean = float(exog[:, lin_idx].mean())
    except Exception:
        # last resort: try to get from original dataframe if stored
        r_mean = None
        try:
            df = res.model.data.frame
            if "StudentTeacherRatio" in df.columns:
                r_mean = float(df["StudentTeacherRatio"].mean())
        except Exception:
            pass
        if r_mean is None:
            raise ValueError("Could not compute mean StudentTeacherRatio from model.exog or model.data.frame")

    # Build linear combination R for the marginal effect: d(E[Y])/d(r) = beta1 + 2*beta2*r
    k_params = len(res.params)
    R = np.zeros((1, k_params))
    R[0, lin_idx] = 1.0
    R[0, quad_idx] = 2.0 * r_mean

    # Use statsmodels t_test to get effect, se, t, p, and conf int (this uses model cov_params)
    try:
        ttest = res.t_test(R)
        me = float(ttest.effect)              # marginal effect at mean
        me_se = float(ttest.sd)               # standard error
        me_t = float(ttest.tvalue)
        # t_test may return pvalue as array-like
        me_p = float(ttest.pvalue) if np.size(ttest.pvalue) == 1 else float(np.array(ttest.pvalue).ravel()[0])
        ci_low, ci_high = tuple(np.asarray(ttest.conf_int()).ravel())
    except Exception:
        # Manual delta-method if t_test is unavailable
        cov = res.cov_params()
        beta = res.params.values if hasattr(res.params, "values") else np.asarray(res.params)
        beta1 = float(beta[lin_idx])
        beta2 = float(beta[quad_idx])
        me = beta1 + 2.0 * beta2 * r_mean
        var_me = (cov.iloc[lin_idx, lin_idx]
                  + (2.0 * r_mean) ** 2 * cov.iloc[quad_idx, quad_idx]
                  + 2.0 * (2.0 * r_mean) * cov.iloc[lin_idx, quad_idx])
        me_se = float(np.sqrt(var_me))
        me_t = me / me_se
        # Two-sided p-value using normal approx (large-sample)
        try:
            from scipy import stats
            me_p = 2.0 * (1.0 - stats.t.cdf(abs(me_t), df=res.df_resid))
        except Exception:
            # normal approximation
            from math import erf, sqrt
            z = abs(me_t)
            me_p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
        ci_low = me - 1.96 * me_se
        ci_high = me + 1.96 * me_se

    # Also extract the separate coefficients and p-values for reference
    params = dict()
    for idx in (lin_idx, quad_idx):
        name = param_names[idx]
        params[name] = {
            "coef": float(res.params.iloc[idx]) if hasattr(res.params, "iloc") else float(res.params[idx]),
            "pvalue": float(res.pvalues.iloc[idx]) if hasattr(res.pvalues, "iloc") else float(res.pvalues[idx])
        }

    # Decision: if marginal effect at mean is negative and statistically significant (p < 0.05),
    # then lower student-teacher ratio (fewer students per teacher) is associated with higher academic performance.
    alpha = 0.05
    if me_p < alpha and me < 0:
        conclusion = ("Yes: a lower student-teacher ratio is associated with higher academic performance. "
                      f"At the sample mean ratio ({r_mean:.3f}), increasing the ratio by 1 student is estimated to change "
                      f"AcademicScore by {me:.3f} points (SE={me_se:.3f}, t={me_t:.3f}, p={me_p:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]).")
    elif me_p < alpha and me > 0:
        conclusion = ("No (opposite): the estimate implies higher student-teacher ratio is associated with higher performance. "
                      f"Marginal effect at mean ({r_mean:.3f}): {me:.3f} (SE={me_se:.3f}, p={me_p:.3g}).")
    else:
        conclusion = ("Ambiguous / not statistically significant: the marginal effect of StudentTeacherRatio at the sample mean "
                      f"is {me:.3f} (SE={me_se:.3f}, t={me_t:.3f}, p={me_p:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
                      "so we cannot confidently conclude that lower ratio is associated with higher academic performance.")

    result_object = {
        "parameters": params,
        "mean_StudentTeacherRatio": r_mean,
        "marginal_effect_at_mean": me,
        "marginal_effect_se": me_se,
        "marginal_effect_t": me_t,
        "marginal_effect_pvalue": me_p,
        "marginal_effect_95CI": (ci_low, ci_high),
        "conclusion": conclusion
    }

    return {"object": result_object,
            "description": ("We compute the marginal effect d(AcademicScore)/d(StudentTeacherRatio) = "
                            "beta1 + 2*beta2*ratio evaluated at the sample mean ratio. The returned object "
                            "contains the coefficients for the linear and quadratic terms, the marginal effect "
                            "at the sample mean, its SE, t-statistic, p-value, 95% CI, and a concise conclusion "
                            "about whether a lower student-teacher ratio is associated with higher academic performance.")}
