def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, confidence intervals, and p-values
    for the beauty effect on teaching evaluations from the provided
    clustered statsmodels results.

    Expects model_output to be a dict with keys:
      - 'base_model_clustered': clustered OLSResults (beauty main effect)
      - 'interaction_model_clustered': clustered OLSResults (beauty x female interaction)

    Returns a dict with:
      - "object": dict containing extracted numeric results for the base model
                  and the interaction model (including computed female-specific effect)
      - "description": brief interpretation of those numbers in context
    """
    import numpy as np
    from scipy import stats

    res = {}

    # Helper: get parameter names/index order from result object
    def param_index(res_obj):
        # Try params.index (pandas)
        params = getattr(res_obj, "params", None)
        if params is not None:
            try:
                return list(params.index)
            except Exception:
                pass
        # Try model.exog_names
        model = getattr(res_obj, "model", None)
        if model is not None:
            try:
                return list(getattr(model, "exog_names", []))
            except Exception:
                pass
        # Try param_names
        try:
            pn = getattr(res_obj, "param_names", None)
            if pn is not None:
                return list(pn)
        except Exception:
            pass
        # Fallback: empty
        return []

    # Helper: get a named value from array-like or dict-like attribute
    def get_named_value(attr, names, name):
        if attr is None:
            return None
        # If pandas Series / dict-like with label
        try:
            if hasattr(attr, "__getitem__") and name in names:
                # If attr supports label indexing (pandas Series)
                try:
                    return float(attr[name])
                except Exception:
                    # Fallback to positional
                    pos = names.index(name)
                    val = attr[pos]
                    return float(val)
        except Exception:
            pass
        # Attr might be ndarray without names
        try:
            if isinstance(attr, (list, tuple, np.ndarray)) and name in names:
                pos = names.index(name)
                return float(attr[pos])
        except Exception:
            pass
        # Attr might be dict-like
        try:
            if isinstance(attr, dict) and name in attr:
                return float(attr[name])
        except Exception:
            pass
        return None

    # Helper to safe-get param info
    def param_info(res_obj, name):
        info = {}
        names = param_index(res_obj)
        params = getattr(res_obj, "params", None)
        bse = getattr(res_obj, "bse", None)
        pvals = getattr(res_obj, "pvalues", None)

        info["coef"] = get_named_value(params, names, name)
        info["se"] = get_named_value(bse, names, name)
        info["p_value"] = get_named_value(pvals, names, name)

        # t-values
        tvals = getattr(res_obj, "tvalues", None)
        info["t"] = get_named_value(tvals, names, name)

        # conf_int: could be DataFrame or ndarray
        try:
            ci = res_obj.conf_int()
            if ci is None:
                raise Exception
            # If DataFrame-like with index
            try:
                # pandas DataFrame
                if hasattr(ci, "loc") and name in param_index(res_obj):
                    row = ci.loc[name]
                    info["ci_lower"] = float(row.iloc[0])
                    info["ci_upper"] = float(row.iloc[1])
                else:
                    # ndarray: use positional mapping
                    names_ci = param_index(res_obj)
                    if name in names_ci:
                        pos = names_ci.index(name)
                        info["ci_lower"] = float(ci[pos, 0])
                        info["ci_upper"] = float(ci[pos, 1])
                    else:
                        info["ci_lower"] = None
                        info["ci_upper"] = None
            except Exception:
                # Try array-like
                names_ci = param_index(res_obj)
                if name in names_ci:
                    pos = names_ci.index(name)
                    info["ci_lower"] = float(ci[pos, 0])
                    info["ci_upper"] = float(ci[pos, 1])
                else:
                    info["ci_lower"] = None
                    info["ci_upper"] = None
        except Exception:
            info["ci_lower"] = None
            info["ci_upper"] = None

        return info

    # Base model: effect of beauty (overall)
    m1 = model_output.get("base_model_clustered")
    base_stats = {}
    if m1 is None:
        raise ValueError("base_model_clustered not found in model_output")

    # Check presence of parameter using param_index
    if "beauty_z" in param_index(m1):
        base_stats = param_info(m1, "beauty_z")
    else:
        raise KeyError("Parameter 'beauty_z' not present in base model results")

    # Interaction model
    m2 = model_output.get("interaction_model_clustered")
    if m2 is None:
        raise ValueError("interaction_model_clustered not found in model_output")

    required_params = ["beauty_z", "beauty_x_female"]
    for p in required_params:
        if p not in param_index(m2):
            raise KeyError(f"Parameter '{p}' not present in interaction model results")

    male_stats = param_info(m2, "beauty_z")
    inter_stats = param_info(m2, "beauty_x_female")

    # Compute female-specific effect: beauty_z + beauty_x_female
    coef_m = male_stats.get("coef")
    coef_int = inter_stats.get("coef")
    cov = None
    try:
        cov_mat = m2.cov_params()
        # cov_mat can be DataFrame or ndarray
        names = param_index(m2)
        if hasattr(cov_mat, "loc"):
            cov_11 = float(cov_mat.loc["beauty_z", "beauty_z"])
            cov_22 = float(cov_mat.loc["beauty_x_female", "beauty_x_female"])
            cov_12 = float(cov_mat.loc["beauty_z", "beauty_x_female"])
            cov = (cov_11, cov_22, cov_12)
        else:
            # ndarray
            if "beauty_z" in names and "beauty_x_female" in names:
                i = names.index("beauty_z")
                j = names.index("beauty_x_female")
                cov_11 = float(cov_mat[i, i])
                cov_22 = float(cov_mat[j, j])
                cov_12 = float(cov_mat[i, j])
                cov = (cov_11, cov_22, cov_12)
    except Exception:
        cov = None

    female_stats = {}
    if coef_m is not None and coef_int is not None:
        female_stats["coef"] = coef_m + coef_int
        # Compute SE for linear combination using covariance matrix if available
        if cov is not None:
            var_f = cov[0] + cov[1] + 2 * cov[2]
            se_f = np.sqrt(var_f) if var_f >= 0 else np.nan
            female_stats["se"] = float(se_f)
            # z/t and p-value: use normal approx
            if female_stats["se"] is not None and not np.isnan(female_stats["se"]) and female_stats["se"] != 0:
                t_f = female_stats["coef"] / female_stats["se"]
                female_stats["t"] = float(t_f)
                female_stats["p_value"] = float(2 * stats.norm.sf(abs(t_f)))
                ci_low = female_stats["coef"] - 1.96 * female_stats["se"]
                ci_high = female_stats["coef"] + 1.96 * female_stats["se"]
                female_stats["ci_lower"] = float(ci_low)
                female_stats["ci_upper"] = float(ci_high)
            else:
                female_stats["t"] = None
                female_stats["p_value"] = None
                female_stats["ci_lower"] = None
                female_stats["ci_upper"] = None
        else:
            # Fallback: if covariance missing, cannot compute SE for sum
            female_stats["se"] = None
            female_stats["t"] = None
            female_stats["p_value"] = None
            female_stats["ci_lower"] = None
            female_stats["ci_upper"] = None
    else:
        female_stats = {k: None for k in ["coef", "se", "t", "p_value", "ci_lower", "ci_upper"]}

    res["base_model"] = base_stats
    res["interaction_model"] = {
        "male_beauty_effect": male_stats,
        "female_beauty_effect": female_stats,
        "interaction_term": inter_stats,
    }

    # Formatting helpers
    def is_num(x):
        return x is not None and (isinstance(x, (int, float)) and not np.isnan(x))

    def fmt(x, digits=4):
        if is_num(x):
            return format(x, f".{digits}f")
        return "NA"

    def fmt_p(x):
        if is_num(x):
            # use 3 significant digits like .3g
            return format(x, ".3g")
        return "NA"

    # Short interpretation string
    def sig_label(p):
        if p is None:
            return "p unknown"
        try:
            return "significant (p < 0.05)" if p < 0.05 else "not significant (p >= 0.05)"
        except Exception:
            return "p unknown"

    desc_lines = []
    # Base model interpretation
    b = base_stats
    desc_lines.append(
        "Base model: one SD increase in standardized beauty (beauty_z) changes eval by "
        f"{fmt(b.get('coef'))} "
        f"(SE={fmt(b.get('se'))}, t={fmt(b.get('t'), digits=3)}, p={fmt_p(b.get('p_value'))}; "
        f"95% CI [{fmt(b.get('ci_lower'))}, {fmt(b.get('ci_upper'))}]) — {sig_label(b.get('p_value'))}."
    )

    # Interaction model interpretation
    male = male_stats
    inter = inter_stats
    fem = female_stats
    desc_lines.append(
        "Interaction model (gender moderator): "
        f"For male instructors (gender_female=0), one SD increase in beauty -> change in eval = {fmt(male.get('coef'))} "
        f"(SE={fmt(male.get('se'))}, p={fmt_p(male.get('p_value'))}) — {sig_label(male.get('p_value'))}. "
        f"Interaction term (beauty_x_female) = {fmt(inter.get('coef'))} (SE={fmt(inter.get('se'))}, p={fmt_p(inter.get('p_value'))}) "
        f"which is the difference in the beauty effect for females vs males — {sig_label(inter.get('p_value'))}. "
        f"Implied female effect = {fmt(fem.get('coef'))} (SE={fmt(fem.get('se'))}), approx p = {fmt_p(fem.get('p_value'))} — {sig_label(fem.get('p_value'))}."
    )

    description = " ".join(desc_lines)

    return {"object": res, "description": description}