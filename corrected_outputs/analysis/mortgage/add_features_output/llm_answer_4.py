def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of applicant gender on mortgage acceptance
    from a fitted statsmodels Logit (BinaryResultsWrapper) object.

    Returns a dictionary with:
      - "object": dict of numeric results (coefficients, p-values, ORs, CIs) for:
          * female (effect when black=0)
          * female x black interaction
          * combined female effect for Black applicants (female + female:black)
      - "description": brief interpretation of what the numbers mean.
    """
    import math
    import numpy as np

    # Try to import a normal CDF (use scipy if available, otherwise use math.erf)
    try:
        from scipy.stats import norm
        norm_cdf = norm.cdf
    except Exception:
        norm_cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object (missing .params)")

    params = res.params
    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame-like with two columns (lower, upper)
    cov = res.cov_params()

    names = [str(n) for n in params.index]

    # Helper to find parameter names robustly
    def find_param(name_substrs):
        """
        Find a parameter name that contains all substrings in name_substrs (case-insensitive).
        Returns the first match or raises if not found.
        """
        lcs = [s.lower() for s in names]
        for i, n in enumerate(lcs):
            if all(sub.lower() in n for sub in name_substrs):
                return names[i]
        raise KeyError(f"Could not find a parameter containing all of: {name_substrs}")

    # Locate parameter names for female and the interaction female x black
    try:
        female_name = find_param(["female"])  # expects exact 'female' or something containing 'female'
    except KeyError:
        raise KeyError("Could not find the 'female' parameter in model_output.params")

    # Interaction: look for a parameter that contains both 'female' and 'black'
    interaction_name = None
    try:
        interaction_name = find_param(["female", "black"])
    except KeyError:
        # It's possible there's no interaction term in the model (but the code specified one).
        interaction_name = None

    # Extract stats for female (effect when black == 0)
    coef_f = float(params[female_name])
    se_f = float(np.sqrt(cov.loc[female_name, female_name])) if female_name in cov.index else None
    p_f = float(pvalues[female_name]) if female_name in pvalues.index else None
    ci_f = tuple(conf.loc[female_name].astype(float)) if female_name in conf.index else None
    or_f = math.exp(coef_f)
    or_f_ci = (math.exp(ci_f[0]), math.exp(ci_f[1])) if ci_f is not None else (None, None)

    result = {
        "female": {
            "param_name": female_name,
            "coef_log_odds": coef_f,
            "se": se_f,
            "p_value": p_f,
            "95%_CI_log_odds": ci_f,
            "odds_ratio": or_f,
            "95%_CI_OR": or_f_ci,
            "meaning": "Effect of being female on log-odds of mortgage acceptance for non-Black applicants (black=0)."
        }
    }

    # Interaction term (female:black) if present
    if interaction_name is not None and interaction_name in params.index:
        coef_int = float(params[interaction_name])
        se_int = float(np.sqrt(cov.loc[interaction_name, interaction_name])) if interaction_name in cov.index else None
        p_int = float(pvalues[interaction_name]) if interaction_name in pvalues.index else None
        ci_int = tuple(conf.loc[interaction_name].astype(float)) if interaction_name in conf.index else None
        or_int = math.exp(coef_int)
        or_int_ci = (math.exp(ci_int[0]), math.exp(ci_int[1])) if ci_int is not None else (None, None)

        result["female_x_black_interaction"] = {
            "param_name": interaction_name,
            "coef_log_odds": coef_int,
            "se": se_int,
            "p_value": p_int,
            "95%_CI_log_odds": ci_int,
            "odds_ratio": or_int,
            "95%_CI_OR": or_int_ci,
            "meaning": "Additional effect on log-odds of mortgage acceptance of being female when applicant is Black (interaction term)."
        }

        # Combined effect for Black applicants: female + interaction
        coef_comb = coef_f + coef_int
        # Var(f + i) = Var(f) + Var(i) + 2*Cov(f,i)
        var_f = float(cov.loc[female_name, female_name])
        var_i = float(cov.loc[interaction_name, interaction_name])
        cov_fi = float(cov.loc[female_name, interaction_name])
        se_comb = math.sqrt(var_f + var_i + 2.0 * cov_fi)
        z_comb = coef_comb / se_comb if se_comb > 0 else float("nan")
        p_comb = 2.0 * (1.0 - norm_cdf(abs(z_comb))) if not math.isnan(z_comb) else None
        ci_comb = (coef_comb - 1.96 * se_comb, coef_comb + 1.96 * se_comb)
        or_comb = math.exp(coef_comb)
        or_comb_ci = (math.exp(ci_comb[0]), math.exp(ci_comb[1]))

        result["female_effect_if_black"] = {
            "formula": "female + (female x black)",
            "coef_log_odds": coef_comb,
            "se": se_comb,
            "z_stat": z_comb,
            "p_value": p_comb,
            "95%_CI_log_odds": ci_comb,
            "odds_ratio": or_comb,
            "95%_CI_OR": or_comb_ci,
            "meaning": "Total effect of being female on log-odds of mortgage acceptance for Black applicants (black=1)."
        }

    else:
        # If no interaction term found, the female effect applies to all (or interaction wasn't estimated)
        result["note"] = "No female x black interaction term found in the model. The 'female' coefficient above applies on average (no subgroup difference estimated)."

    # Optionally include model-level goodness-of-fit or summary string for reference
    try:
        summary_text = str(res.summary())
        result["model_summary_text_snippet"] = summary_text[:4000]  # truncate large output
    except Exception:
        result["model_summary_text_snippet"] = None

    # Build short description
    if interaction_name is not None and interaction_name in params.index:
        description = (
            "Extracted: (1) 'female' coefficient (effect on log-odds for non-Black applicants), "
            "(2) 'female x black' interaction coefficient (additional effect for Black applicants), "
            "and (3) combined female effect for Black applicants (female + interaction). "
            "Reported coefficients, SEs, p-values, 95% CIs (log-odds) and odds ratios with CIs. "
            "Interpret OR < 1 means lower odds of acceptance for females compared to males; OR > 1 means higher odds."
        )
    else:
        description = (
            "Extracted: 'female' coefficient (effect on log-odds of acceptance). No female x black interaction found. "
            "Reported coefficient, SE, p-value, 95% CI (log-odds) and odds ratio with CI. "
            "Interpret OR < 1 means lower odds of acceptance for females compared to males; OR > 1 means higher odds."
        )

    return {"object": result, "description": description}