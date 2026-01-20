def extract_final_answer(model_output):
    """
    Extracts the gender effect (female) from a fitted statsmodels GLM/Logit result,
    including the interaction female:black if present, and returns coefficients,
    standard errors, z-stats, p-values, 95% CIs, and odds ratios for:
      - Female effect among non-Black applicants (black=0)
      - Female effect among Black applicants (black=1) [if interaction present]
    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Plain-language interpretation of the numbers"
      }
    """
    import numpy as np
    import math

    res = model_output

    # Retrieve parameters, covariance matrix, and confidence intervals
    params = res.params  # pandas Series
    cov = res.cov_params()  # DataFrame
    conf = res.conf_int(alpha=0.05)  # DataFrame with 0 and 1 columns

    names = list(params.index)

    # Helper: find parameter name robustly
    def find_name_containing(token, exclude_tokens=None):
        exclude_tokens = exclude_tokens or []
        for n in names:
            if token in n and not any(ex in n for ex in exclude_tokens):
                return n
        return None

    # Find female main effect name (prefer exact 'female' if present)
    female_name = 'female' if 'female' in names else find_name_containing('female', exclude_tokens=[':','*','black','Black'])
    # Find interaction term name that includes both female and black
    interaction_name = None
    for n in names:
        if 'female' in n and 'black' in n:
            interaction_name = n
            break

    # Ensure female term exists
    if female_name is None:
        raise KeyError("Could not find a parameter representing the main effect of 'female' in model parameters: "
                       f"found names = {names}")

    # Extract female main effect (this is effect when black == 0)
    coef_f = float(params[female_name])
    se_f = float(res.bse[female_name]) if hasattr(res, 'bse') else float(np.sqrt(cov.loc[female_name, female_name]))
    ci_f = tuple(conf.loc[female_name])
    z_f = coef_f / se_f if se_f != 0 else float('nan')
    # two-sided p-value from normal approx:
    cdf = 0.5 * (1 + math.erf(abs(z_f) / math.sqrt(2)))
    p_f = 2 * (1 - cdf)

    # Odds ratio and CI
    or_f = float(np.exp(coef_f))
    or_ci_f = (float(np.exp(ci_f[0])), float(np.exp(ci_f[1])))

    result = {
        'female_non_black': {
            'param_name': female_name,
            'coef_log_odds': coef_f,
            'se': se_f,
            'z': z_f,
            'p_value': p_f,
            '95%_CI_log_odds': ci_f,
            'odds_ratio': or_f,
            '95%_CI_odds_ratio': or_ci_f
        }
    }

    description_lines = []
    description_lines.append(
        "The reported 'female' coefficient is the effect of being female on the log-odds of approval for non-Black applicants (black=0)."
    )

    # If interaction present, compute combined effect for Black applicants using delta method
    if interaction_name is not None and interaction_name in params.index:
        coef_int = float(params[interaction_name])
        # Combined effect = female + female:black
        coef_comb = coef_f + coef_int

        # Variance of sum: Var(f) + Var(int) + 2*Cov(f,int)
        var_f = float(cov.loc[female_name, female_name])
        var_int = float(cov.loc[interaction_name, interaction_name])
        cov_f_int = float(cov.loc[female_name, interaction_name])
        se_comb = math.sqrt(var_f + var_int + 2 * cov_f_int)

        ci_low = coef_comb - 1.96 * se_comb
        ci_high = coef_comb + 1.96 * se_comb
        z_comb = coef_comb / se_comb if se_comb != 0 else float('nan')
        cdf_comb = 0.5 * (1 + math.erf(abs(z_comb) / math.sqrt(2)))
        p_comb = 2 * (1 - cdf_comb)

        or_comb = float(np.exp(coef_comb))
        or_ci_comb = (float(np.exp(ci_low)), float(np.exp(ci_high)))

        result['female_black'] = {
            'param_name': f"{female_name} + {interaction_name}",
            'coef_log_odds': coef_comb,
            'se': se_comb,
            'z': z_comb,
            'p_value': p_comb,
            '95%_CI_log_odds': (ci_low, ci_high),
            'odds_ratio': or_comb,
            '95%_CI_odds_ratio': or_ci_comb,
            'female_coef': coef_f,
            'interaction_coef': coef_int
        }

        description_lines.append(
            "The female effect for Black applicants (black=1) is the sum of the 'female' coefficient and the 'female:black' interaction coefficient. "
            "I report that combined log-odds coef, its SE (via the delta method), z, p-value, 95% CI, and the corresponding odds ratio and CI."
        )

    else:
        description_lines.append("No female:black interaction term was found in the model parameters; only the main female effect (for non-Black) is reported.")

    description_lines.append(
        "Interpretation: coefficients are log-odds. Exponentiated coefficients are odds ratios. "
        "A positive coefficient (odds ratio > 1) means being female increases the odds of approval; "
        "a negative coefficient (odds ratio < 1) means being female decreases the odds of approval. "
        "P-values indicate whether these effects are statistically distinguishable from zero."
    )

    description = " ".join(description_lines)

    return {"object": result, "description": description}