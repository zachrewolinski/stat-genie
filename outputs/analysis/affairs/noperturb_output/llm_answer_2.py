def extract_final_answer(model_output):
    """
    Extracts the estimated effect of having children on reported extramarital affairs
    from a fitted statsmodels GLM (Negative Binomial) object that used the formula:
      AffairCount ~ HasChildren + Gender_Male + HasChildren:Gender_Male + ...
    Returns a dict with numerical results under "object" and a short human-readable
    interpretation under "description".
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    # Helper to format results for one coefficient
    def summarize_effect(beta, se):
        z = beta / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else np.nan
        ci_low = beta - 1.96 * se
        ci_upp = beta + 1.96 * se
        irr = np.exp(beta)
        irr_ci = (np.exp(ci_low), np.exp(ci_upp))
        pct_change = (irr - 1.0) * 100.0
        return {
            "coef_log": float(beta),
            "se": float(se),
            "z": float(z),
            "p_value": float(p),
            "irr": float(irr),
            "irr_95ci": (float(irr_ci[0]), float(irr_ci[1])),
            "percent_change": float(pct_change),
            "log_95ci": (float(ci_low), float(ci_upp))
        }

    # Validate object has expected attributes
    if not hasattr(model_output, "params") or not hasattr(model_output, "cov_params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object with .params and .cov_params()")

    params = model_output.params
    cov = model_output.cov_params()

    # Names of terms as expected from the formula
    term_base = "HasChildren"
    term_inter = "HasChildren:Gender_Male"  # statsmodels uses ':' for interaction in formulas

    if term_base not in params.index:
        raise KeyError(f"Expected coefficient '{term_base}' not found in model params: {list(params.index)}")

    beta_base = float(params[term_base])
    var_base = float(cov.loc[term_base, term_base])
    se_base = sqrt(var_base)

    # Summarize baseline effect (this is effect for Gender_Male == 0, i.e., females under coding)
    effect_female = summarize_effect(beta_base, se_base)

    # Prepare male effect = HasChildren + HasChildren:Gender_Male (if interaction present)
    if term_inter in params.index:
        beta_inter = float(params[term_inter])
        # variance of sum = var(b1) + var(b2) + 2*cov(b1,b2)
        var_inter = float(cov.loc[term_inter, term_inter])
        cov_b1b2 = float(cov.loc[term_base, term_inter])
        beta_male = beta_base + beta_inter
        var_male = var_base + var_inter + 2.0 * cov_b1b2
        se_male = sqrt(max(var_male, 0.0))
        effect_male = summarize_effect(beta_male, se_male)
        interaction_present = True
    else:
        # No interaction term: effect is same for males and females
        effect_male = effect_female.copy()
        interaction_present = False

    # Also extract individual coefficient p-values/CIs for reporting
    # For completeness include raw HasChildren and interaction c.oefs if present
    raw = {}
    raw["HasChildren"] = {
        "coef_log": float(beta_base),
        "se": float(se_base),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(beta_base / se_base)))) if se_base > 0 else np.nan,
        "irr": float(np.exp(beta_base)),
        "irr_95ci": (float(np.exp(beta_base - 1.96 * se_base)), float(np.exp(beta_base + 1.96 * se_base))),
    }
    if interaction_present:
        beta_inter = float(params[term_inter])
        se_inter = sqrt(float(cov.loc[term_inter, term_inter]))
        raw["HasChildren:Gender_Male"] = {
            "coef_log": float(beta_inter),
            "se": float(se_inter),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(beta_inter / se_inter)))) if se_inter > 0 else np.nan,
            "irr": float(np.exp(beta_inter)),
            "irr_95ci": (float(np.exp(beta_inter - 1.96 * se_inter)), float(np.exp(beta_inter + 1.96 * se_inter))),
        }

    # Decision rules for short conclusion
    def interpret(effect_dict):
        p = effect_dict["p_value"]
        irr = effect_dict["irr"]
        pct = effect_dict["percent_change"]
        if np.isnan(p):
            return "Effect could not be tested (missing SE/p)."
        if p < 0.05:
            if irr < 1.0:
                return f"Statistically significant decrease: expected affair count {abs(pct):.1f}% lower (IRR={irr:.3f}, p={p:.3g})."
            else:
                return f"Statistically significant increase: expected affair count {pct:.1f}% higher (IRR={irr:.3f}, p={p:.3g})."
        else:
            return f"No statistically significant effect detected (IRR={irr:.3f}, p={p:.3g})."

    conclusion_female = interpret(effect_female)
    conclusion_male = interpret(effect_male)

    # Build return object
    result_object = {
        "effect_female": effect_female,
        "effect_male": effect_male,
        "interaction_present": interaction_present,
        "raw_terms": raw,
        "model_summary_available": hasattr(model_output, "summary")  # quick indicator
    }

    # Short human-readable description
    if interaction_present:
        description = (
            "Estimated effect of having children on reported extramarital affairs (Negative Binomial GLM, log link):\n"
            f"- Females (Gender_Male=0): {conclusion_female}\n"
            f"- Males (Gender_Male=1): {conclusion_male}\n"
            "Interpretation: IRR < 1 indicates fewer reported affairs associated with having children; IRR > 1 indicates more. "
            "See 'object' for coefficients, standard errors, p-values, IRRs, and 95% CIs."
        )
    else:
        description = (
            "Estimated (common) effect of having children on reported extramarital affairs (no HasChildren:Gender_Male interaction found):\n"
            f"- All respondents: {conclusion_female}\n"
            "Interpretation: IRR < 1 indicates fewer reported affairs associated with having children; IRR > 1 indicates more. "
            "See 'object' for coefficients, standard errors, p-values, IRRs, and 95% CIs."
        )

    return {"object": result_object, "description": description}