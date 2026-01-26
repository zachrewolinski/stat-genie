def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    marginal effects for the beauty variable from the provided statsmodels
    OLSResults objects (cluster-robust results).

    model_output: dict-like with keys 'model1' and 'model2' containing fitted
                  statsmodels regression result objects (the clustered robust results).

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Brief interpretation of the results in context."
      }

    The returned "object" contains:
      - model1: coef, se, pvalue, 95% CI for beauty_z (simple bivariate)
      - model2: coef, se, pvalue, 95% CI for beauty_z and beauty_z_sq (full model)
      - marginal_effects: estimated marginal effect of beauty (d eval / d beauty)
          at beauty = -1, 0, +1 (i.e., -1SD, mean, +1SD), plus SE and 95% CI
      - joint_test: Wald test (joint hypothesis beauty_z = 0 and beauty_z_sq = 0)
          returning test statistic and p-value (if available)
    """
    import numpy as np

    def _params_and_cov(res):
        """
        Return (names_list, params_array, cov_array)
        - names_list: list of parameter names in order
        - params_array: numpy array of parameter values in same order
        - cov_array: numpy 2D array of covariance matrix in same order
        """
        # params
        params = getattr(res, "params", None)
        if params is None:
            raise AttributeError("Result object has no 'params' attribute.")
        # Determine names and values
        if hasattr(params, "index"):
            names = list(params.index)
            params_arr = np.asarray(params)
        else:
            # params is likely an ndarray; try to get names from model.exog_names
            if hasattr(res, "model") and hasattr(res.model, "exog_names"):
                names = list(res.model.exog_names)
            else:
                # fallback: create generic names
                params_arr = np.asarray(params)
                names = [f"param{i}" for i in range(len(params_arr))]
            params_arr = np.asarray(params)

        # covariance
        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = getattr(res, "cov_params_default", None)
            if cov is None:
                # try attribute 'normalized_cov_params' or '_cov_params' as last resorts
                cov = getattr(res, "normalized_cov_params", None) or getattr(res, "_cov_params", None)
        if cov is None:
            raise AttributeError("Could not obtain covariance matrix from result object.")
        cov_arr = np.asarray(cov)

        return names, params_arr, cov_arr

    def _get_attr_value(res, attr, name, names):
        """
        Safely retrieve an attribute (like pvalues or bse) for a named parameter.
        Returns np.nan if not available.
        """
        arr = getattr(res, attr, None)
        if arr is None:
            return float("nan")
        if hasattr(arr, "index"):
            # pandas Series-like
            try:
                return float(arr[name])
            except Exception:
                return float("nan")
        else:
            # assume ndarray aligned with names
            try:
                idx = names.index(name)
            except ValueError:
                return float("nan")
            arr_np = np.asarray(arr)
            try:
                return float(arr_np[idx])
            except Exception:
                return float("nan")

    # Validate input
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict with keys 'model1' and 'model2'.")

    if 'model1' not in model_output or 'model2' not in model_output:
        raise KeyError("model_output must contain keys 'model1' and 'model2'.")

    res1 = model_output['model1']
    res2 = model_output['model2']

    # Extract for model1
    names1, params1_arr, cov1_arr = _params_and_cov(res1)
    var = 'beauty_z'
    if var not in names1:
        raise KeyError(f"Variable '{var}' not found in model1 results parameters: {names1}")
    i_var1 = names1.index(var)
    coef1 = float(params1_arr[i_var1])
    # standard error
    se1 = _get_attr_value(res1, "bse", var, names1)
    if np.isnan(se1):
        # fallback to sqrt of cov diag
        try:
            se1 = float(np.sqrt(cov1_arr[i_var1, i_var1]))
        except Exception:
            se1 = float("nan")
    # p-value
    p1 = _get_attr_value(res1, "pvalues", var, names1)
    ci1 = [coef1 - 1.96 * se1, coef1 + 1.96 * se1]

    # Extract for model2
    names2, params2_arr, cov2_arr = _params_and_cov(res2)
    for v in ['beauty_z', 'beauty_z_sq']:
        if v not in names2:
            raise KeyError(f"Variable '{v}' not found in model2 results parameters: {names2}")

    i_b = names2.index('beauty_z')
    i_b2 = names2.index('beauty_z_sq')
    coef_b = float(params2_arr[i_b])
    coef_b2 = float(params2_arr[i_b2])

    se_b = _get_attr_value(res2, "bse", 'beauty_z', names2)
    se_b2 = _get_attr_value(res2, "bse", 'beauty_z_sq', names2)
    # fallback to covariance diag if needed
    if np.isnan(se_b):
        try:
            se_b = float(np.sqrt(cov2_arr[i_b, i_b]))
        except Exception:
            se_b = float("nan")
    if np.isnan(se_b2):
        try:
            se_b2 = float(np.sqrt(cov2_arr[i_b2, i_b2]))
        except Exception:
            se_b2 = float("nan")

    p_b = _get_attr_value(res2, "pvalues", 'beauty_z', names2)
    p_b2 = _get_attr_value(res2, "pvalues", 'beauty_z_sq', names2)
    ci_b = [coef_b - 1.96 * se_b, coef_b + 1.96 * se_b]
    ci_b2 = [coef_b2 - 1.96 * se_b2, coef_b2 + 1.96 * se_b2]

    # Marginal effects: derivative = beta1 + 2*beta2 * beauty
    cov_sub = cov2_arr[[i_b, i_b2], :][:, [i_b, i_b2]]  # 2x2

    def marginal_effect_at(bval):
        a = np.array([1.0, 2.0 * bval])
        est = coef_b + 2.0 * coef_b2 * bval
        try:
            var_me = float(a @ cov_sub @ a.T)
            se_me = float(np.sqrt(var_me)) if var_me >= 0 else float("nan")
        except Exception:
            var_me = float("nan")
            se_me = float("nan")
        ci_low = est - 1.96 * se_me
        ci_high = est + 1.96 * se_me
        return {"beauty": float(bval), "marginal_effect": float(est), "se": se_me, "95ci": [ci_low, ci_high]}

    me_minus1 = marginal_effect_at(-1.0)
    me_0 = marginal_effect_at(0.0)
    me_plus1 = marginal_effect_at(1.0)

    # Joint test: H0: beauty_z = 0 and beauty_z_sq = 0
    k = len(names2)
    R = np.zeros((2, k))
    R[0, i_b] = 1.0
    R[1, i_b2] = 1.0
    try:
        wtest = res2.wald_test(R)
        w_stat = None
        w_pvalue = None
        # statistic
        stat_attr = None
        for attr in ("statistic", "fvalue", "chi2", "stat"):
            if hasattr(wtest, attr):
                stat_attr = getattr(wtest, attr)
                break
        if stat_attr is not None:
            stat_arr = np.asarray(stat_attr)
            try:
                w_stat = float(stat_arr.item())
            except Exception:
                # if not scalar, convert to float of first element
                try:
                    w_stat = float(stat_arr.flatten()[0])
                except Exception:
                    w_stat = None
        # pvalue
        p_attr = None
        for attr in ("pvalue", "prob", "p_f", "p"):
            if hasattr(wtest, attr):
                p_attr = getattr(wtest, attr)
                break
        if p_attr is not None:
            p_arr = np.asarray(p_attr)
            try:
                w_pvalue = float(p_arr.item())
            except Exception:
                try:
                    w_pvalue = float(p_arr.flatten()[0])
                except Exception:
                    w_pvalue = None
        joint_test = {"statistic": w_stat, "pvalue": w_pvalue}
    except Exception as e:
        joint_test = {"error": f"joint test failed: {str(e)}"}

    # Interpretation
    signif1 = (not np.isnan(p1)) and (p1 < 0.05)
    signif_b = (not np.isnan(p_b)) and (p_b < 0.05)
    signif_b2 = (not np.isnan(p_b2)) and (p_b2 < 0.05)

    if signif_b or signif_b2:
        effect_evidence = "There is statistical evidence that instructor beauty is associated with student evaluations in the full model."
    else:
        effect_evidence = "There is no statistically significant evidence in the full model that instructor beauty is associated with student evaluations at conventional levels (p < 0.05)."

    if signif_b2:
        nonlin_note = "The squared term is significant, suggesting a nonlinear (quadratic) relationship."
    else:
        nonlin_note = "The squared term is not significant, providing no strong evidence of nonlinearity."

    # Safely format p-values/se/ci for description even if nan
    def _fmt(x, nd=4):
        try:
            if np.isnan(x):
                return "NA"
            return f"{x:.{nd}f}"
        except Exception:
            return str(x)

    description = (
        f"Model 1 (bivariate): beauty_z coef = {_fmt(coef1)}, SE = {_fmt(se1)}, p = {_fmt(p1,3)}. "
        f"Model 2 (controls + quadratic): beauty_z coef = {_fmt(coef_b)} (p = {_fmt(p_b,3)}), "
        f"beauty_z_sq coef = {_fmt(coef_b2)} (p = {_fmt(p_b2,3)}). "
        f"{effect_evidence} {nonlin_note} "
        "Marginal effects (d eval / d beauty) are provided at -1SD, 0, +1SD in the 'object' field."
    )

    output_object = {
        "model1": {
            "variable": "beauty_z",
            "coef": coef1,
            "se": se1,
            "pvalue": p1,
            "95ci": ci1
        },
        "model2": {
            "beauty_z": {
                "coef": coef_b,
                "se": se_b,
                "pvalue": p_b,
                "95ci": ci_b
            },
            "beauty_z_sq": {
                "coef": coef_b2,
                "se": se_b2,
                "pvalue": p_b2,
                "95ci": ci_b2
            },
            "marginal_effects": {
                "at_-1": me_minus1,
                "at_0": me_0,
                "at_1": me_plus1
            },
            "joint_test_beauty_and_sq": joint_test
        }
    }

    return {"object": output_object, "description": description}