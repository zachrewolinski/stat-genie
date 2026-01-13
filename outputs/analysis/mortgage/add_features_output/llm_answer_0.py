def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and odds-ratio
    summaries for the gender effect from a fitted statsmodels GLMResultsWrapper (logistic).
    
    Returns:
      {
        "object": {
            "female_nonblack": {coef, se, pvalue, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper},
            "female_black":    {coef, se, pvalue, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper},
            "interaction":     {name, coef, se, pvalue, ci_lower, ci_upper, OR, OR_ci_lower, OR_ci_upper},
            "notes": {"female_param_name", "interaction_param_name", "method_for_combined_pvalue"}
        },
        "description": "Text explaining the numbers and their meaning in context."
      }
    """
    import numpy as np
    import statsmodels.api as sm

    res = model_output

    # Parameter names available in the fitted model
    param_names = list(res.params.index)

    # Heuristics to find main female effect (a name containing 'female' but NOT 'black')
    female_candidates = [n for n in param_names if ('female' in n.lower() and 'black' not in n.lower())]
    if not female_candidates:
        raise KeyError(f"Could not find a main 'female' parameter in model params: {param_names}")
    # Prefer exact 'female' if present
    female_param = 'female' if 'female' in param_names else sorted(female_candidates, key=len)[0]

    # Find interaction: a name that contains both 'female' and 'black'
    interaction_candidates = [n for n in param_names if ('female' in n.lower() and 'black' in n.lower())]
    interaction_param = interaction_candidates[0] if interaction_candidates else None

    # Helper to extract confint row robustly (works if conf_int returns ndarray or DataFrame)
    conf = res.conf_int()
    def get_conf(name):
        try:
            # If conf is a DataFrame or ndarray with matching index
            return np.array(conf.loc[name])
        except Exception:
            # fallback: find index position
            pos = param_names.index(name)
            return np.array(conf[pos])

    # Collect main female effect (this is effect among non-Black because of interaction)
    coef_f = float(res.params[female_param])
    se_f = float(res.bse[female_param])
    p_f = float(res.pvalues[female_param])
    ci_f = get_conf(female_param)
    or_f = float(np.exp(coef_f))
    or_ci_f = np.exp(ci_f.astype(float))

    female_nonblack = {
        "param_name": female_param,
        "coef_log_odds": coef_f,
        "se": se_f,
        "p_value": p_f,
        "ci_95_lower": float(ci_f[0]),
        "ci_95_upper": float(ci_f[1]),
        "odds_ratio": or_f,
        "odds_ratio_ci_95_lower": float(or_ci_f[0]),
        "odds_ratio_ci_95_upper": float(or_ci_f[1]),
    }

    # Interaction term summary (if present)
    if interaction_param is not None:
        coef_int = float(res.params[interaction_param])
        se_int = float(res.bse[interaction_param])
        p_int = float(res.pvalues[interaction_param])
        ci_int = get_conf(interaction_param)
        or_int = float(np.exp(coef_int))
        or_ci_int = np.exp(ci_int.astype(float))

        interaction = {
            "param_name": interaction_param,
            "coef_log_odds": coef_int,
            "se": se_int,
            "p_value": p_int,
            "ci_95_lower": float(ci_int[0]),
            "ci_95_upper": float(ci_int[1]),
            "odds_ratio": or_int,
            "odds_ratio_ci_95_lower": float(or_ci_int[0]),
            "odds_ratio_ci_95_upper": float(or_ci_int[1]),
        }
    else:
        interaction = None

    # Compute combined effect for Black applicants: female + (female:black)
    if interaction is not None:
        # combined coef
        coef_fb = coef_f + float(res.params[interaction_param])
        # variance of sum using covariance matrix
        cov = res.cov_params()
        try:
            var_fb = cov.loc[female_param, female_param] \
                   + cov.loc[interaction_param, interaction_param] \
                   + 2 * cov.loc[female_param, interaction_param]
        except Exception:
            # fallback to array indexing if cov is ndarray
            idx_f = param_names.index(female_param)
            idx_int = param_names.index(interaction_param)
            var_fb = cov[idx_f, idx_f] + cov[idx_int, idx_int] + 2 * cov[idx_f, idx_int]
        se_fb = float(np.sqrt(var_fb))
        # z-stat and p-value (Wald test normal approx)
        z_fb = coef_fb / se_fb if se_fb > 0 else np.nan
        from scipy import stats
        p_fb = float(2 * (1 - stats.norm.cdf(abs(z_fb))))
        # 95% CI for combined effect
        ci_fb = np.array([coef_fb - 1.96 * se_fb, coef_fb + 1.96 * se_fb])
        or_fb = float(np.exp(coef_fb))
        or_ci_fb = np.exp(ci_fb.astype(float))

        female_black = {
            "param_name_combined": f"{female_param} + {interaction_param}",
            "coef_log_odds": float(coef_fb),
            "se": se_fb,
            "z_value": float(z_fb),
            "p_value": p_fb,
            "ci_95_lower": float(ci_fb[0]),
            "ci_95_upper": float(ci_fb[1]),
            "odds_ratio": or_fb,
            "odds_ratio_ci_95_lower": float(or_ci_fb[0]),
            "odds_ratio_ci_95_upper": float(or_ci_fb[1]),
        }

        # Also compute a formal test for combined hypothesis = 0 using model's t_test (Wald)
        # This yields the same p-value in large samples but uses model machinery.
        try:
            # construct linear restriction string like "female + female:black = 0"
            restriction = f"{female_param} + {interaction_param} = 0"
            ttest_res = res.t_test(restriction)
            combined_p_via_ttest = float(ttest_res.pvalue)
        except Exception:
            combined_p_via_ttest = None
    else:
        female_black = None
        combined_p_via_ttest = None

    output_object = {
        "female_nonblack": female_nonblack,
        "female_black": female_black,
        "interaction": interaction,
        "notes": {
            "female_param_name": female_param,
            "interaction_param_name": interaction_param,
            "method_for_combined_pvalue": "Wald z (computed) and model.t_test (if available)",
            "combined_pvalue_via_t_test": combined_p_via_ttest
        }
    }

    # Short human-readable description
    if interaction is not None:
        description = (
            "This output summarizes the gender effect on mortgage acceptance from a logistic GLM that "
            "includes an interaction between female and Black. 'female_nonblack' is the log-odds "
            "difference (and odds ratio) for female vs male among non-Black applicants (this is the "
            "coefficient on the female main effect). 'female_black' is the combined effect for Black "
            "applicants (female main effect + female:black interaction) with its standard error, "
            "Wald z and p-value, and odds-ratio. 'interaction' reports the interaction coefficient "
            "itself (tests whether the gender effect differs for Black applicants). Use the p-values "
            "to judge statistical significance (typical threshold 0.05)."
        )
    else:
        description = (
            "This model summary gives the female main effect (log-odds and odds ratio) but no "
            "female:black interaction term was found. The female effect reported is the difference "
            "in log-odds (and odds ratio) for female vs male applicants (across the sample)."
        )

    return {"object": output_object, "description": description}