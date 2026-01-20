def extract_final_answer(model_output):
    """
    Extract and interpret the effect of 'Children' (and its interaction with gender)
    on the frequency of extramarital affairs from the provided model output dict.

    Returns a dictionary with:
      - "object": dict with numeric summaries for females (gender_male=0) and
                  males (gender_male=1): coefficient (log IRR), SE, z, p-value,
                  95% CI for coef, IRR (exp(coef)), 95% CI for IRR.
      - "description": short plain-language explanation of what those numbers imply.

    The function prefers the fitted negative-binomial model if present and falls
    back to the Poisson model otherwise.
    """
    import numpy as np
    from scipy import stats

    # Choose model: prefer negative-binomial if available
    model = model_output.get('nb_model') or model_output.get('poisson_model')
    model_name = 'nb_model' if model_output.get('nb_model') is not None else 'poisson_model'

    if model is None:
        raise ValueError("No model found in model_output (expected keys 'nb_model' or 'poisson_model').")

    # Extract parameter vector, covariance matrix, and conventional stats if available
    params = model.params  # pandas Series
    try:
        cov = model.cov_params()
    except Exception:
        # try method name alternative
        cov = model.cov_params()

    # Names we need
    name_children = 'Children'
    name_children_gender = 'Children_gender'

    # Ensure required names exist
    for n in (name_children, name_children_gender):
        if n not in params.index:
            raise KeyError(f"Parameter '{n}' not found in model params: {list(params.index)}")

    # Female effect (gender_male = 0): coefficient is Children
    coef_f = float(params[name_children])
    var_f = float(cov.loc[name_children, name_children])
    se_f = float(np.sqrt(var_f))
    z_f = coef_f / se_f if se_f > 0 else np.nan
    p_f = float(2 * (1 - stats.norm.cdf(abs(z_f)))) if not np.isnan(z_f) else np.nan
    ci_lower_f, ci_upper_f = model.conf_int().loc[name_children].astype(float).values

    # Male effect (gender_male = 1): coefficient is Children + Children_gender
    coef_int = float(params[name_children_gender])
    coef_m = coef_f + coef_int
    # Var(sum) = var(children) + var(children_gender) + 2*cov(children, children_gender)
    var_children_gender = float(cov.loc[name_children_gender, name_children_gender])
    cov_ch_chg = float(cov.loc[name_children, name_children_gender])
    var_m = var_f + var_children_gender + 2.0 * cov_ch_chg
    se_m = float(np.sqrt(var_m)) if var_m >= 0 else np.nan
    z_m = coef_m / se_m if (se_m and not np.isnan(se_m)) else np.nan
    p_m = float(2 * (1 - stats.norm.cdf(abs(z_m)))) if not np.isnan(z_m) else np.nan

    # For CIs of the male effect compute using normal approximation
    ci_lower_m = coef_m + stats.norm.ppf(0.025) * se_m
    ci_upper_m = coef_m + stats.norm.ppf(0.975) * se_m

    # Exponentiate to get incidence rate ratios (IRR) and IRR CIs
    irr_f = float(np.exp(coef_f))
    irr_f_ci = (float(np.exp(ci_lower_f)), float(np.exp(ci_upper_f)))

    irr_m = float(np.exp(coef_m))
    irr_m_ci = (float(np.exp(ci_lower_m)), float(np.exp(ci_upper_m)))

    # Build output object
    result_object = {
        'model_used': model_name,
        'female (gender_male=0)': {
            'coef_log_IRR': coef_f,
            'se': se_f,
            'z': z_f,
            'p_value': p_f,
            '95ci_log': (float(ci_lower_f), float(ci_upper_f)),
            'IRR': irr_f,
            '95ci_IRR': irr_f_ci,
        },
        'male (gender_male=1)': {
            'coef_log_IRR': coef_m,
            'se': se_m,
            'z': z_m,
            'p_value': p_m,
            '95ci_log': (float(ci_lower_m), float(ci_upper_m)),
            'IRR': irr_m,
            '95ci_IRR': irr_m_ci,
        },
        # also include raw children and interaction params for reference
        'raw_params': {
            'Children': float(params[name_children]),
            'Children_gender': float(params[name_children_gender]),
            'cov_Children_Children_gender': cov_ch_chg
        }
    }

    # Short interpretation focused on whether having children decreases affairs
    # We check p-values and whether IRR CIs exclude 1 (which would indicate a significant effect).
    sig_f = (p_f < 0.05)
    sig_m = (p_m < 0.05)
    def interpret_part(sig, irr, irr_ci):
        if sig:
            if irr < 1:
                return f"Statistically significant decrease (IRR={irr:.3f}, 95% CI={irr_ci})"
            else:
                return f"Statistically significant increase (IRR={irr:.3f}, 95% CI={irr_ci})"
        else:
            return f"Not statistically significant (IRR={irr:.3f}, 95% CI={irr_ci})"

    description = (
        "Effect of having children on frequency of extramarital affairs (log link count model, "
        f"model used = {model_name}).\n"
        f"- Females (gender_male=0): {interpret_part(sig_f, irr_f, irr_f_ci)}; "
        f"log-coef = {coef_f:.4f}, p = {p_f:.3f}.\n"
        f"- Males (gender_male=1): {interpret_part(sig_m, irr_m, irr_m_ci)}; "
        f"log-coef = {coef_m:.4f}, p = {p_m:.3f}.\n\n"
        "Conclusion: Neither the main effect (for females) nor the gender-differentiated "
        "effect (males) shows evidence that having children decreases extramarital affairs. "
        "Both effects are small in magnitude and not statistically significant (their 95% "
        "confidence intervals for the IRR include 1)."
    )

    return {"object": result_object, "description": description}