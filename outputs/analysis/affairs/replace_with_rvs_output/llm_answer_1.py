def extract_final_answer(model_output):
    """
    Extract the estimated effect of "Children" on "affairs" from the fitted Tobit model,
    including the interaction with Female. Returns a dictionary with keys:
      - "object": nested dict with estimates, standard errors, z-stats, p-values, and 95% CIs
                  for the effect of Children for males and for females, plus raw params.
      - "description": short plain-English interpretation of the results in context.

    Assumptions about parameter ordering (matches the model construction in the prompt):
      params vector = [const, Children, Female, Children_x_Female, Age, YearsMarried,
                       Religiousness, Education, Occupation, Rating, log_sigma]
    The function is robust to model objects that store cov_params() as ndarray or DataFrame.
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output

    # Attempt to get params and covariance matrix
    params = np.asarray(res.params)  # length = k + 1 (last element = log_sigma)
    try:
        cov = res.cov_params()
        cov = np.asarray(cov)
    except Exception:
        # If cov_params is not available, try to use bse (approximate)
        cov = None

    # Define expected parameter names/order used when fitting the model
    exog_cols = [
        'Children',
        'Female',
        'Children_x_Female',
        'Age',
        'YearsMarried',
        'Religiousness',
        'Education',
        'Occupation',
        'Rating'
    ]
    names = ['const'] + exog_cols  # params[0] corresponds to const

    # Check length consistency
    k = len(names)
    if params.shape[0] < k + 1:
        raise ValueError("Unexpected number of parameters in model_output.params. "
                         f"Found {params.shape[0]}, expected at least {k+1} (including log_sigma).")

    # indices
    idx_children = names.index('Children')
    idx_inter = names.index('Children_x_Female')

    # Extract raw coefficients (latent Tobit coefficients)
    beta_children = float(params[idx_children])
    beta_inter = float(params[idx_inter])

    # Prepare results container
    out = {
        'raw_params': {
            'Children_coef': beta_children,
            'Children_x_Female_coef': beta_inter,
            'log_sigma': float(params[k])  # last param is log_sigma
        }
    }

    # If covariance matrix available, compute SEs and test statistics for linear combinations
    if cov is not None and cov.shape[0] >= k + 1:
        var_children = float(cov[idx_children, idx_children])
        var_inter = float(cov[idx_inter, idx_inter])
        cov_child_inter = float(cov[idx_children, idx_inter])

        # Effect of Children for males (Female=0): just beta_children
        eff_male = beta_children
        se_male = np.sqrt(var_children) if var_children >= 0 else np.nan
        z_male = eff_male / se_male if se_male and not np.isnan(se_male) else np.nan
        p_male = 2.0 * (1.0 - norm.cdf(abs(z_male))) if not np.isnan(z_male) else np.nan
        ci_male = (eff_male - 1.96 * se_male, eff_male + 1.96 * se_male) if not np.isnan(se_male) else (np.nan, np.nan)

        # Effect of Children for females (Female=1): beta_children + beta_inter
        eff_female = beta_children + beta_inter
        var_female = var_children + var_inter + 2.0 * cov_child_inter
        se_female = np.sqrt(var_female) if var_female >= 0 else np.nan
        z_female = eff_female / se_female if se_female and not np.isnan(se_female) else np.nan
        p_female = 2.0 * (1.0 - norm.cdf(abs(z_female))) if not np.isnan(z_female) else np.nan
        ci_female = (eff_female - 1.96 * se_female, eff_female + 1.96 * se_female) if not np.isnan(se_female) else (np.nan, np.nan)

        out['effects'] = {
            'male (Female=0)': {
                'effect': eff_male,
                'se': se_male,
                'z': z_male,
                'p_value': p_male,
                '95%_CI': ci_male,
                'interpretation': ("Coefficient on latent Tobit scale: negative => having children "
                                   "is associated with lower latent propensity for affairs")
            },
            'female (Female=1)': {
                'effect': eff_female,
                'se': se_female,
                'z': z_female,
                'p_value': p_female,
                '95%_CI': ci_female,
                'interpretation': ("Combined coefficient (Children + Children_x_Female) on latent "
                                   "Tobit scale for females")
            }
        }
    else:
        # If covariance unavailable, at least return raw coefficients and note inability to test significance
        out['effects'] = {
            'male (Female=0)': {
                'effect': beta_children,
                'se': None,
                'z': None,
                'p_value': None,
                '95%_CI': (None, None),
                'note': 'covariance matrix not available; cannot compute SE/p-values'
            },
            'female (Female=1)': {
                'effect': beta_children + beta_inter,
                'se': None,
                'z': None,
                'p_value': None,
                '95%_CI': (None, None),
                'note': 'covariance matrix not available; cannot compute SE/p-values'
            }
        }

    # Short textual summary
    # We interpret negative effects as "having children decreases engagement in extramarital affairs"
    def interpret(eff, p):
        if p is None or np.isnan(p):
            return "Coefficient is {0:.4f}. Statistical significance could not be determined.".format(eff)
        if p < 0.001:
            sig = "highly statistically significant (p < 0.001)"
        elif p < 0.01:
            sig = "statistically significant (p < 0.01)"
        elif p < 0.05:
            sig = "statistically significant (p < 0.05)"
        else:
            sig = "not statistically significant (p >= 0.05)"
        direction = "decrease" if eff < 0 else ("increase" if eff > 0 else "no change")
        return "Estimated effect = {0:.4f}, which implies a {1} in the latent propensity for affairs. {2} (p = {3:.4g}).".format(eff, direction, sig, p)

    male_summary = interpret(out['effects']['male (Female=0)']['effect'],
                             out['effects']['male (Female=0)']['p_value'])
    female_summary = interpret(out['effects']['female (Female=1)']['effect'],
                               out['effects']['female (Female=1)']['p_value'])

    description = (
        "This Tobit model reports latent-scale effects of having children on the number of extramarital "
        "affairs (left-censored at 0). Interpretation is on the latent Tobit scale (not directly the observed "
        "expected count). Summary:\n"
        f"- Males (Female=0): {male_summary}\n"
        f"- Females (Female=1): {female_summary}\n\n"
        "Negative coefficients indicate that having children is associated with a lower latent propensity to have "
        "affairs. Use the provided p-values and confidence intervals (if available) to judge statistical significance."
    )

    return {"object": out, "description": description}