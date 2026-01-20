def extract_final_answer(model_output):
    """
    Extracts the effect of 'female' on mortgage application acceptance from a fitted
    statsmodels Logit/GLM binary results object.
    
    Returns a dictionary with:
      - "object": a dict containing numeric results (coefficients, SEs, p-values,
                  95% CIs, odds ratios and OR CIs) for:
            * female effect for non-Black applicants (the main effect)
            * female effect for Black applicants (if a female_black_interaction
              term is present: main + interaction)
      - "description": a short interpretation of the reported numbers and how to read them.
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output  # statsmodels results object

    # Fetch parameter estimates, standard errors, p-values, and covariance matrix
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    cov = res.cov_params()

    # z for 95% CI
    z95 = norm.ppf(0.975)

    def safe_get(arr, name):
        """Helper to get value from Series-like object, raises useful error if missing."""
        if name not in arr.index:
            raise KeyError(f"Model output does not contain a parameter named '{name}'.")
        return arr[name]

    # Ensure 'female' is present
    try:
        beta_f = safe_get(params, 'female')
        se_f = safe_get(bse, 'female')
        p_f = safe_get(pvalues, 'female')
    except KeyError as e:
        # Return an informative object if female is missing
        return {
            "object": None,
            "description": f"Model output does not contain required parameter 'female'. Error: {e}"
        }

    # Compute 95% CI for female main effect (log-odds scale)
    ci_f = (beta_f - z95 * se_f, beta_f + z95 * se_f)
    # Odds ratio and its CI
    or_f = float(np.exp(beta_f))
    or_ci_f = (float(np.exp(ci_f[0])), float(np.exp(ci_f[1])))

    results = {
        "female_nonblack": {
            "coef_log_odds": float(beta_f),
            "se": float(se_f),
            "p_value": float(p_f),
            "95ci_log_odds": (float(ci_f[0]), float(ci_f[1])),
            "odds_ratio": or_f,
            "95ci_odds_ratio": or_ci_f
        }
    }

    # If interaction term 'female_black_interaction' present, compute female effect among Black applicants
    interaction_name = 'female_black_interaction'
    if interaction_name in params.index:
        beta_int = safe_get(params, interaction_name)
        # Compute combined effect: female + interaction (i.e., effect of female when black=1)
        beta_f_black = beta_f + beta_int

        # Variance: Var(f) + Var(int) + 2*Cov(f,int)
        try:
            var_f = cov.loc['female', 'female']
            var_int = cov.loc[interaction_name, interaction_name]
            cov_f_int = cov.loc['female', interaction_name]
        except Exception:
            # If cov is ndarray, map indices
            idx = list(params.index)
            i_f = idx.index('female')
            i_int = idx.index(interaction_name)
            var_f = float(cov[i_f, i_f])
            var_int = float(cov[i_int, i_int])
            cov_f_int = float(cov[i_f, i_int])

        var_f_black = var_f + var_int + 2.0 * cov_f_int
        se_f_black = float(np.sqrt(max(var_f_black, 0.0)))  # guard against tiny negative numerical noise
        z_f_black = beta_f_black / se_f_black if se_f_black > 0 else np.nan
        p_f_black = float(2.0 * (1.0 - norm.cdf(abs(z_f_black)))) if se_f_black > 0 else float('nan')
        ci_f_black = (beta_f_black - z95 * se_f_black, beta_f_black + z95 * se_f_black)
        or_f_black = float(np.exp(beta_f_black))
        or_ci_f_black = (float(np.exp(ci_f_black[0])), float(np.exp(ci_f_black[1])))

        results["female_black"] = {
            "coef_log_odds": float(beta_f_black),
            "se": se_f_black,
            "p_value": p_f_black,
            "95ci_log_odds": (float(ci_f_black[0]), float(ci_f_black[1])),
            "odds_ratio": or_f_black,
            "95ci_odds_ratio": or_ci_f_black,
            "notes": "This is the marginal effect of being female among Black applicants (female + female:black interaction)."
        }

    # Build a human-readable description explaining how to interpret the numbers
    description_lines = [
        "Returned metrics show the estimated effect of being female on the log-odds of mortgage acceptance,",
        "plus corresponding standard errors, two-sided p-values, 95% confidence intervals (on the log-odds scale),",
        "and the exponentiated values (odds ratios) with their 95% CIs.",
        "Interpretation rules:",
        "- Positive coef_log_odds (or odds_ratio > 1) means females have higher odds of acceptance relative to males.",
        "- Negative coef_log_odds (or odds_ratio < 1) means females have lower odds of acceptance relative to males.",
        "- p_value < 0.05 indicates statistical significance at the 5% level for that effect.",
    ]
    if "female_black" in results:
        description_lines.append(
            "- Because a female_black interaction term is present, results include the female effect among non-Black applicants "
            "(the main 'female' effect) and the female effect among Black applicants (female + interaction)."
        )

    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }