def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of HasChildren on count of affairs
    from a fitted statsmodels GLM (Negative Binomial) that includes an
    interaction term HasChildren_Gender (HasChildren * Gender_Male).

    Returns a dictionary with:
      - "object": dict containing coefficients, standard errors, p-values,
                  95% CIs, incidence rate ratios (IRR) and IRR CIs for:
                    * HasChildren effect for females (reference, Gender_Male=0)
                    * HasChildren effect for males (sum of main + interaction)
                    * the interaction coefficient itself
      - "description": a short plain-language interpretation of whether having
                       children decreases engagement in extramarital affairs,
                       separately for females and males, based on the estimates.
    """
    import math
    import numpy as np

    res = model_output

    # Required parameter names
    main_name = 'HasChildren'
    inter_name = 'HasChildren_Gender'
    gender_name = 'Gender_Male'

    # Basic checks
    params = getattr(res, 'params', None)
    if params is None:
        raise ValueError("model_output has no .params attribute; not a statsmodels results object")

    for name in (main_name, inter_name):
        if name not in params.index:
            raise KeyError(f"Required parameter '{name}' not found in model results. Found params: {list(params.index)}")

    # Extract main and interaction coefficients and SEs/pvals/CIs
    coef_main = float(params[main_name])
    coef_inter = float(params[inter_name])

    bse = res.bse
    se_main = float(bse[main_name])
    se_inter = float(bse[inter_name])

    pvals = res.pvalues
    p_main = float(pvals[main_name])
    p_inter = float(pvals[inter_name])

    conf = res.conf_int()  # DataFrame or ndarray-like with rows matching params.index
    # conf intervals for main and interaction
    ci_main = tuple(map(float, conf.loc[main_name].values)) if hasattr(conf, 'loc') else tuple(map(float, conf[params.index.get_loc(main_name)]))
    ci_inter = tuple(map(float, conf.loc[inter_name].values)) if hasattr(conf, 'loc') else tuple(map(float, conf[params.index.get_loc(inter_name)]))

    # Compute effect for males: sum of main + interaction
    coef_male = coef_main + coef_inter

    # Compute SE for the sum using covariance matrix
    cov = res.cov_params()
    # Ensure cov has the needed entries
    if main_name not in cov.index or inter_name not in cov.index:
        raise KeyError("Covariance matrix missing required rows/cols for HasChildren and HasChildren_Gender")
    var_main = float(cov.loc[main_name, main_name])
    var_inter = float(cov.loc[inter_name, inter_name])
    cov_main_inter = float(cov.loc[main_name, inter_name])
    se_male = math.sqrt(var_main + var_inter + 2.0 * cov_main_inter)

    # z -> two-sided p-values using standard normal distribution via erf
    def two_sided_p_from_z(z):
        # standard normal CDF
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return 2.0 * (1.0 - cdf) if z >= 0 else 2.0 * cdf

    z_main = coef_main / se_main if se_main > 0 else float('nan')
    z_inter = coef_inter / se_inter if se_inter > 0 else float('nan')
    z_male = coef_male / se_male if se_male > 0 else float('nan')

    p_main_fromz = two_sided_p_from_z(abs(z_main))
    p_inter_fromz = two_sided_p_from_z(abs(z_inter))
    p_male = two_sided_p_from_z(abs(z_male))

    # 95% CI for male coefficient
    ci_male = (coef_male - 1.96 * se_male, coef_male + 1.96 * se_male)

    # Convert log-coef to incidence rate ratios (IRR) and CIs
    irr_main = float(np.exp(coef_main))
    irr_inter = float(np.exp(coef_inter))
    irr_male = float(np.exp(coef_male))
    irr_ci_main = (float(np.exp(ci_main[0])), float(np.exp(ci_main[1])))
    irr_ci_inter = (float(np.exp(ci_inter[0])), float(np.exp(ci_inter[1])))
    irr_ci_male = (float(np.exp(ci_male[0])), float(np.exp(ci_male[1])))

    # Prepare output object
    out_obj = {
        'HasChildren (female; Gender_Male=0)': {
            'coef_log': coef_main,
            'se': se_main,
            'z': z_main,
            'p_value': p_main,                 # model p-value (should match p_main_fromz)
            'p_value_from_z': p_main_fromz,
            '95%_CI_log': (ci_main[0], ci_main[1]),
            'IRR': irr_main,
            '95%_CI_IRR': irr_ci_main,
            'interpretation': (
                "Effect of having children for females (reference group). "
                "IRR < 1 means fewer expected affairs; IRR > 1 means more expected affairs."
            )
        },
        'HasChildren (male)': {
            'coef_log': coef_male,
            'se': se_male,
            'z': z_male,
            'p_value': p_male,
            '95%_CI_log': (ci_male[0], ci_male[1]),
            'IRR': irr_male,
            '95%_CI_IRR': irr_ci_male,
            'interpretation': (
                "Effect of having children for males (Gender_Male=1): main + interaction."
            )
        },
        'HasChildren_Gender (interaction term)': {
            'coef_log': coef_inter,
            'se': se_inter,
            'z': z_inter,
            'p_value': p_inter,
            '95%_CI_log': (ci_inter[0], ci_inter[1]),
            'IRR': irr_inter,
            '95%_CI_IRR': irr_ci_inter,
            'interpretation': (
                "Additional effect of HasChildren when respondent is male (difference between males and females)."
            )
        },
        'notes': {
            'model_family': getattr(res.model.family, '__class__', str(type(res.model.family))).__name__ if hasattr(res, 'model') else None,
            'link_function': getattr(getattr(res.model, 'family', None), 'link', None).__class__.__name__ if hasattr(res, 'model') and getattr(res.model, 'family', None) is not None else None,
            'interpretation_note': (
                "Coefficients are on the log count scale (log link). "
                "IRR = exp(coef) is multiplicative effect on expected count of affairs. "
                "Because the model includes an interaction HasChildren_Gender, the main "
                "HasChildren coefficient represents the effect for the reference gender (Gender_Male=0, i.e., females). "
                "The effect for males equals main + interaction."
            )
        }
    }

    # Compose a short human-readable conclusion
    def sig_label(p):
        return "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p >= 0.05)"

    concl_lines = []
    # Female conclusion
    fem_dir = "decrease" if irr_main < 1 else ("increase" if irr_main > 1 else "no change")
    concl_lines.append(
        f"For females (Gender_Male=0): HasChildren coef = {coef_main:.4f}, IRR = {irr_main:.3f} "
        f"(95% CI IRR: [{irr_ci_main[0]:.3f}, {irr_ci_main[1]:.3f}]), {sig_label(p_main_fromz)}. "
        f"Interpretation: having children is associated with a {100*(1-irr_main):.1f}% {fem_dir} in expected count of affairs."
    )
    # Male conclusion
    male_dir = "decrease" if irr_male < 1 else ("increase" if irr_male > 1 else "no change")
    concl_lines.append(
        f"For males (Gender_Male=1): HasChildren effect (main + interaction) coef = {coef_male:.4f}, IRR = {irr_male:.3f} "
        f"(95% CI IRR: [{irr_ci_male[0]:.3f}, {irr_ci_male[1]:.3f}]), {sig_label(p_male)}. "
        f"Interpretation: having children is associated with a {100*(1-irr_male):.1f}% {male_dir} in expected count of affairs for males."
    )
    # Interaction significance
    concl_lines.append(
        f"The interaction term HasChildren_Gender has coef = {coef_inter:.4f}, IRR = {irr_inter:.3f}, "
        f"95% CI IRR: [{irr_ci_inter[0]:.3f}, {irr_ci_inter[1]:.3f}], {sig_label(p_inter_fromz)}. "
        f"If the interaction is statistically significant, it indicates the effect of children differs by gender."
    )

    description = " ".join(concl_lines)

    return {"object": out_obj, "description": description}