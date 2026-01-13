def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-scores, p-values, and 95% CIs for:
      - age_c (linear age effect)
      - age_sq (quadratic age effect)
      - each age_c:culture interaction term (if present)
    Also computes per-culture age slopes (reference culture = age_c; other cultures =
    age_c + corresponding interaction), their SEs, z-scores, p-values, and 95% CIs.
    Finally performs a joint Wald test of whether all age_c:culture interaction
    coefficients are simultaneously zero (i.e., whether developmental slopes differ
    across cultures).

    Returns a dictionary:
      {
        "object": { ... detailed numeric results ... },
        "description": "Brief interpretation of results in context."
      }
    """
    import re
    import numpy as np
    from scipy.stats import norm

    res = model_output  # statsmodels GLMResultsWrapper

    params = res.params.copy()
    pvalues = res.pvalues.copy()
    try:
        conf = res.conf_int().copy()
    except Exception:
        # conf_int sometimes returns ndarray; convert to DataFrame-like for indexing
        ci_arr = res.conf_int()
        conf = {}
        for i, name in enumerate(params.index):
            conf[name] = (float(ci_arr[i, 0]), float(ci_arr[i, 1]))

    cov = res.cov_params()

    out = {}
    # Helper to format single-parameter summaries
    def summarize_param(name):
        coef = float(params[name])
        se = float(np.sqrt(cov.loc[name, name]))
        z = coef / se if se > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
        if isinstance(conf, dict):
            ci_low, ci_high = conf[name]
        else:
            ci_low, ci_high = float(conf.loc[name, 0]), float(conf.loc[name, 1])
        return {"coef": coef, "se": se, "z": z, "p": p, "95% CI": [ci_low, ci_high]}

    # Ensure age_c and age_sq exist
    summary = {}
    if 'age_c' in params.index:
        summary['age_c'] = summarize_param('age_c')
    else:
        raise KeyError("Model does not contain 'age_c' parameter")

    if 'age_sq' in params.index:
        summary['age_sq'] = summarize_param('age_sq')
    else:
        # not fatal, but note absent
        summary['age_sq'] = None

    # Find interaction parameters for age_c by culture.
    # Robust matching: any param name that contains 'age_c' AND ('C(culture)' or 'culture')
    interaction_names = [name for name in params.index
                         if ('age_c' in name) and ('culture' in name) and (name != 'age_c')]

    # Summarize each interaction parameter individually
    interactions_summary = {}
    for name in interaction_names:
        interactions_summary[name] = summarize_param(name)

    # Compute per-culture age slopes:
    # Reference culture (the baseline) slope is the 'age_c' coefficient.
    # For each interaction term, slope = age_c + interaction_coef
    culture_slopes = {}
    # reference
    ref_slope = float(params['age_c'])
    ref_se = float(np.sqrt(cov.loc['age_c', 'age_c']))
    ref_z = ref_slope / ref_se if ref_se > 0 else np.nan
    ref_p = float(2 * (1 - norm.cdf(abs(ref_z)))) if not np.isnan(ref_z) else np.nan
    ref_ci = [float(conf.loc['age_c', 0]) if not isinstance(conf, dict) else conf['age_c'][0],
              float(conf.loc['age_c', 1]) if not isinstance(conf, dict) else conf['age_c'][1]]
    culture_slopes["reference"] = {
        "slope_coef": ref_slope, "se": ref_se, "z": ref_z, "p": ref_p, "95% CI": ref_ci,
        "note": "Reference (omitted) culture slope for age"
    }

    for name in interaction_names:
        inter_coef = float(params[name])
        # slope = age_c + interaction
        slope = ref_slope + inter_coef
        # SE(slope) = sqrt(Var(age_c) + Var(inter) + 2*Cov(age_c, inter))
        var_age = float(cov.loc['age_c', 'age_c'])
        var_inter = float(cov.loc[name, name])
        cov_ai = float(cov.loc['age_c', name])
        se_slope = float(np.sqrt(var_age + var_inter + 2 * cov_ai))
        z_slope = slope / se_slope if se_slope > 0 else np.nan
        p_slope = float(2 * (1 - norm.cdf(abs(z_slope)))) if not np.isnan(z_slope) else np.nan
        # 95% CI
        ci_low = slope - 1.96 * se_slope
        ci_high = slope + 1.96 * se_slope

        # try to extract culture label from parameter name like 'age_c:C(culture)[T.2]'
        m = re.search(r'\[T\.?([^\]]+)\]', name)
        label = m.group(1) if m else name

        culture_slopes[label] = {
            "slope_coef": slope,
            "se": se_slope,
            "z": z_slope,
            "p": p_slope,
            "95% CI": [ci_low, ci_high],
            "param_name_for_interaction": name
        }

    # Joint test: are all age_c:culture interactions = 0?
    joint_test = None
    if len(interaction_names) > 0:
        # Build constraint string like "age_c:C(culture)[T.2] = 0, age_c:C(culture)[T.3] = 0"
        constraint = ", ".join([f"{name} = 0" for name in interaction_names])
        try:
            wt = res.wald_test(constraint)
            # wt may have attributes statistic and pvalue
            stat = float(wt.statistic) if hasattr(wt, 'statistic') else float(wt.statistic[0])
            pval = float(wt.pvalue) if hasattr(wt, 'pvalue') else float(wt.p_value if hasattr(wt, 'p_value') else np.nan)
            df = int(wt.df_denom) if hasattr(wt, 'df_denom') else (int(wt.df) if hasattr(wt, 'df') else len(interaction_names))
            joint_test = {"constraint": constraint, "statistic": stat, "pvalue": pval, "df": df}
        except Exception as e:
            joint_test = {"error": f"Failed to run Wald test: {e}", "constraint": constraint}
    else:
        joint_test = {"note": "No age_c:culture interaction parameters found; joint test not applicable."}

    out["params_summary"] = summary
    out["interactions"] = interactions_summary
    out["culture_slopes"] = culture_slopes
    out["interaction_joint_test"] = joint_test

    # Short human-readable interpretation
    # Basic guidance: positive slope => increasing reliance on majority with age; negative => decreasing.
    interpretation_lines = []
    interpretation_lines.append(
        "age_c (linear): coef = {coef:.3f}, p = {p:.3g}. Positive implies increasing reliance on the majority with age "
        "in the reference culture.".format(**{
            "coef": summary['age_c']['coef'],
            "p": summary['age_c']['p']
        })
    )
    if summary.get('age_sq') is not None:
        interpretation_lines.append(
            "age_sq (quadratic): coef = {coef:.3f}, p = {p:.3g}. This captures curvature in the age trajectory.".format(
                **{"coef": summary['age_sq']['coef'] if summary['age_sq'] else np.nan,
                   "p": summary['age_sq']['p'] if summary['age_sq'] else np.nan}
            )
        )
    if len(interaction_names) > 0:
        if isinstance(joint_test, dict) and 'pvalue' in joint_test:
            if joint_test['pvalue'] < 0.05:
                interpretation_lines.append(
                    f"The set of age-by-culture interaction terms is significant (Wald p = {joint_test['pvalue']:.3g}), "
                    "indicating that age-related slopes differ across cultural contexts."
                )
            else:
                interpretation_lines.append(
                    f"The set of age-by-culture interaction terms is NOT significant (Wald p = {joint_test['pvalue']:.3g}), "
                    "indicating no strong evidence that age-related slopes differ across cultures."
                )
        else:
            interpretation_lines.append("Could not compute a joint test for the interactions; inspect individual interaction terms below.")

        interpretation_lines.append("Per-culture slopes (reference and differences): see 'object' -> 'culture_slopes' for numeric estimates.")
    else:
        interpretation_lines.append("No age-by-culture interactions were present in the model: comparable slopes across cultures were assumed in the model.")
    description = " ".join(interpretation_lines)

    return {"object": out, "description": description}