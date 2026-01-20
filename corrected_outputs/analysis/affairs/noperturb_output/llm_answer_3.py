def extract_final_answer(model_output):
    """
    Extracts statistics from a fitted statsmodels GLMResults (Negative Binomial) object
    to answer whether having children decreases engagement in extramarital affairs,
    including moderation by gender.

    Returns a dictionary with:
      - "object": nested dict of numeric results (coefficients, SEs, p-values, CIs, IRRs)
      - "description": plain-language interpretation of those statistics regarding the task
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Basic sanity checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object with .params")

    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        conf = res.conf_int()  # DataFrame: columns [lower, upper]
    except Exception:
        conf = None
    cov = res.cov_params()

    # Helper to safely extract values and convert to python float
    def _get(name, series):
        if name not in series.index:
            raise KeyError(f"Model does not contain parameter '{name}'")
        return float(series[name])

    def _get_conf(name):
        if conf is None or name not in conf.index:
            return [None, None]
        lo, hi = conf.loc[name].values
        return [float(lo), float(hi)]

    # Required parameter names
    names = ["HasChildren", "HasChildren_Female", "Female"]
    for n in names:
        if n not in params.index:
            raise KeyError(f"Expected parameter '{n}' not found in model. Found params: {list(params.index)}")

    # Extract main pieces
    coef_hc = _get("HasChildren", params)
    se_hc = _get("HasChildren", bse)
    p_hc = _get("HasChildren", pvals)
    ci_hc = _get_conf("HasChildren")
    irr_hc = float(np.exp(coef_hc))
    irr_ci_hc = [float(np.exp(ci_hc[0])), float(np.exp(ci_hc[1]))] if None not in ci_hc else [None, None]

    coef_int = _get("HasChildren_Female", params)
    se_int = _get("HasChildren_Female", bse)
    p_int = _get("HasChildren_Female", pvals)
    ci_int = _get_conf("HasChildren_Female")
    irr_int = float(np.exp(coef_int))
    irr_ci_int = [float(np.exp(ci_int[0])), float(np.exp(ci_int[1]))] if None not in ci_int else [None, None]

    # Effect of HasChildren for males (Female=0): just coef_hc
    coef_m = coef_hc
    se_m = se_hc
    p_m = p_hc
    ci_m = ci_hc
    irr_m = irr_hc
    irr_ci_m = irr_ci_hc

    # Effect of HasChildren for females (Female=1): coef_hc + coef_int
    coef_f = coef_hc + coef_int

    # Compute SE for linear combination using covariance matrix
    # Build selector vector a such that effect = a' * params
    # Ensure order of params is respected
    param_names = list(params.index)
    a = np.zeros(len(param_names))
    # set 1 for HasChildren
    a[param_names.index("HasChildren")] = 1.0
    # set 1 for HasChildren_Female
    a[param_names.index("HasChildren_Female")] = 1.0

    var_f = float(a @ cov.values @ a)
    se_f = float(np.sqrt(var_f))
    z_f = coef_f / se_f if se_f != 0 else np.nan
    p_f = float(2 * (1 - stats.norm.cdf(abs(z_f)))) if not np.isnan(z_f) else None
    ci_f = [float(coef_f - 1.96 * se_f), float(coef_f + 1.96 * se_f)]
    irr_f = float(np.exp(coef_f))
    irr_ci_f = [float(np.exp(ci_f[0])), float(np.exp(ci_f[1]))]

    # Interaction interpretation: does HasChildren effect differ by gender?
    coef_interaction = coef_int
    se_interaction = se_int
    p_interaction = p_int
    ci_interaction = ci_int
    irr_interaction = irr_int
    irr_ci_interaction = irr_ci_int

    # Build results object (all floats so serializable)
    results_object = {
        "HasChildren_coefficient": coef_hc,
        "HasChildren_se": se_hc,
        "HasChildren_pvalue": p_hc,
        "HasChildren_95CI": ci_hc,
        "HasChildren_IRR": irr_hc,
        "HasChildren_IRR_95CI": irr_ci_hc,
        "HasChildren_Female_interaction_coefficient": coef_interaction,
        "HasChildren_Female_interaction_se": se_interaction,
        "HasChildren_Female_interaction_pvalue": p_interaction,
        "HasChildren_Female_interaction_95CI": ci_interaction,
        "HasChildren_Female_interaction_IRR": irr_interaction,
        "HasChildren_Female_interaction_IRR_95CI": irr_ci_interaction,
        "Effect_HasChildren_male_coefficient": coef_m,
        "Effect_HasChildren_male_se": se_m,
        "Effect_HasChildren_male_pvalue": p_m,
        "Effect_HasChildren_male_95CI": ci_m,
        "Effect_HasChildren_male_IRR": irr_m,
        "Effect_HasChildren_male_IRR_95CI": irr_ci_m,
        "Effect_HasChildren_female_coefficient": coef_f,
        "Effect_HasChildren_female_se": se_f,
        "Effect_HasChildren_female_pvalue": p_f,
        "Effect_HasChildren_female_95CI": ci_f,
        "Effect_HasChildren_female_IRR": irr_f,
        "Effect_HasChildren_female_IRR_95CI": irr_ci_f,
    }

    # Short plain-language interpretation tailored to the question
    def interpret_effect(coef, pval, irr):
        if pval is None:
            return "Estimate: {:.4f} (p unavailable). IRR={:.4f}".format(coef, irr)
        sig = pval < 0.05
        direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
        if sig:
            pct = (1 - irr) * 100
            if coef < 0:
                return "Statistically significant {} in expected affair count: IRR={:.3f} => about {:.1f}% lower expected count (p={:.3g}).".format(direction, irr, abs(pct), pval)
            else:
                return "Statistically significant {} in expected affair count: IRR={:.3f} => about {:.1f}% higher expected count (p={:.3g}).".format(direction, irr, abs(pct), pval)
        else:
            return "No statistically significant effect (coef={:.4f}, IRR={:.3f}, p={:.3g}).".format(coef, irr, pval)

    interp_male = interpret_effect(coef_m, p_m, irr_m)
    interp_female = interpret_effect(coef_f, p_f, irr_f)
    interp_interaction = ("Interaction term p={:.3g}: indicates the HasChildren effect differs by gender."
                          if p_interaction < 0.05 else
                          "Interaction term p={:.3g}: no strong evidence that the HasChildren effect differs by gender.").format(p_interaction)

    description_lines = [
        "Extracted statistics related to the effect of having children on count of extramarital affairs (Negative Binomial GLM).",
        "",
        "Main coefficient (HasChildren):",
        f" - coef = {coef_hc:.4f}, se = {se_hc:.4f}, p = {p_hc:.4g}",
        f" - 95% CI = [{ci_hc[0]:.4f}, {ci_hc[1]:.4f}]",
        f" - IRR = {irr_hc:.4f} (95% CI [{irr_ci_hc[0]:.4f}, {irr_ci_hc[1]:.4f}])",
        "",
        "Interaction (HasChildren x Female):",
        f" - coef = {coef_interaction:.4f}, se = {se_interaction:.4f}, p = {p_interaction:.4g}",
        f" - 95% CI = [{ci_interaction[0]:.4f}, {ci_interaction[1]:.4f}]",
        f" - IRR = {irr_interaction:.4f} (95% CI [{irr_ci_interaction[0]:.4f}, {irr_ci_interaction[1]:.4f}])",
        "",
        "Gender-specific effects (linear combination results):",
        " - Males (Female=0): " + interp_male,
        " - Females (Female=1): " + interp_female,
        "",
        "Interaction interpretation: " + interp_interaction,
        "",
        "How to use these numbers to answer the question:",
        " - If IRR < 1 and p < 0.05 for a group, having children is associated with a statistically significant decrease in expected affair counts for that group.",
        " - If the interaction is significant (p < 0.05), the HasChildren effect differs by gender; otherwise the effect is not convincingly different across genders.",
    ]

    description = "\n".join(description_lines)

    return {"object": results_object, "description": description}