def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of instructor beauty on course evaluations
    from a fitted statsmodels RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": a dict containing coefficients, SEs, t-stats, p-values, 95% CIs for:
            * beauty_z (main effect)
            * beauty_z_sq (if present)
            * beauty_z_x_gender_male (interaction, if present)
        and marginal effects of beauty for female (gender_male=0) and male (gender_male=1)
        instructors (estimates, SEs, p-values, 95% CIs).
      - "description": a short plain-language interpretation of the main results.

    Notes:
      - Coefficients are in outcome scale (evaluation points; eval is 1-5).
      - beauty_z is standardized (1 unit = 1 SD in attractiveness).
      - Marginal effect for females = coef(beauty_z).
        Marginal effect for males = coef(beauty_z) + coef(beauty_z_x_gender_male).
      - Confidence intervals and p-values use the covariance matrix returned by the model
        (this will reflect cluster-robust SEs if the model was fit with clustering).
    """
    import numpy as np
    from math import isfinite

    # Try to import t distribution; fallback to normal if not available
    try:
        from scipy.stats import t as tdist
        from scipy.stats import norm as normdist
    except Exception:
        tdist = None
        from math import erf, sqrt
        # simple normal functions if scipy not present
        class _NormApprox:
            @staticmethod
            def cdf(x):
                return 0.5 * (1.0 + erf(x / sqrt(2.0)))
            @staticmethod
            def ppf(q):
                # approximate inverse CDF is not provided here; but we won't reach ppf if no scipy
                raise RuntimeError("scipy not available for ppf")
        normdist = _NormApprox()

    res = model_output

    # Basic outputs
    try:
        params = res.params
        cov = res.cov_params()
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object: " + str(e))

    # Helper to build stats for a single coefficient name
    def coef_stats(name):
        if name not in params.index:
            return None
        est = float(params.loc[name])
        # covariance matrix index names may match params.index
        try:
            var = float(cov.loc[name, name])
        except Exception:
            # fallback: try positional indexing
            idx = list(params.index).index(name)
            var = float(cov.iloc[idx, idx])
        se = float(np.sqrt(var)) if var >= 0 else float('nan')
        # degrees of freedom for t; fallback to None
        df = getattr(res, 'df_resid', None)
        t_stat = float(est / se) if (isfinite(est) and isfinite(se) and se != 0) else float('nan')

        # p-value and critical value for 95% CI
        if df is None or (isinstance(df, (int, float)) and df <= 0) or tdist is None:
            # normal approx
            p_val = 2.0 * (1.0 - normdist.cdf(abs(t_stat)))
            # use normal critical z
            try:
                crit = normdist.ppf(0.975)
            except Exception:
                # numeric value for 97.5% z
                crit = 1.959963984540054
        else:
            p_val = 2.0 * (1.0 - tdist.cdf(abs(t_stat), df))
            crit = tdist.ppf(0.975, df)

        ci_lower = est - crit * se
        ci_upper = est + crit * se

        return {
            "name": name,
            "est": est,
            "se": se,
            "t": t_stat,
            "p": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        }

    # Extract main pieces
    beauty = coef_stats('beauty_z')
    beauty_sq = coef_stats('beauty_z_sq')
    interaction = coef_stats('beauty_z_x_gender_male')

    # Prepare marginal effects for female (gender_male=0) and male (gender_male=1)
    marg_female = None
    marg_male = None
    if beauty is not None:
        # female effect = beauty coefficient
        marg_female = {
            "group": "female (gender_male=0)",
            "est": beauty["est"],
            "se": beauty["se"],
            "p": beauty["p"],
            "ci_lower": beauty["ci_lower"],
            "ci_upper": beauty["ci_upper"],
            "note": "Change in eval (points) per 1 SD increase in beauty for female instructors"
        }

    if beauty is not None and interaction is not None:
        # male effect = beauty + interaction
        est_m = beauty["est"] + interaction["est"]
        # compute variance using covariance matrix
        try:
            cov_b_b = cov.loc['beauty_z','beauty_z']
            cov_i_i = cov.loc['beauty_z_x_gender_male','beauty_z_x_gender_male']
            cov_b_i = cov.loc['beauty_z','beauty_z_x_gender_male']
        except Exception:
            # fallback to positional
            idx_b = list(params.index).index('beauty_z')
            idx_i = list(params.index).index('beauty_z_x_gender_male')
            cov_b_b = cov.iloc[idx_b, idx_b]
            cov_i_i = cov.iloc[idx_i, idx_i]
            cov_b_i = cov.iloc[idx_b, idx_i]
        var_m = cov_b_b + cov_i_i + 2.0 * cov_b_i
        se_m = float(np.sqrt(var_m)) if var_m >= 0 else float('nan')
        df = getattr(res, 'df_resid', None)
        t_stat_m = est_m / se_m if (isfinite(est_m) and isfinite(se_m) and se_m != 0) else float('nan')
        if df is None or (isinstance(df, (int, float)) and df <= 0) or tdist is None:
            p_m = 2.0 * (1.0 - normdist.cdf(abs(t_stat_m)))
            try:
                crit = normdist.ppf(0.975)
            except Exception:
                crit = 1.959963984540054
        else:
            p_m = 2.0 * (1.0 - tdist.cdf(abs(t_stat_m), df))
            crit = tdist.ppf(0.975, df)
        ci_lower_m = est_m - crit * se_m
        ci_upper_m = est_m + crit * se_m

        marg_male = {
            "group": "male (gender_male=1)",
            "est": float(est_m),
            "se": float(se_m),
            "p": float(p_m),
            "ci_lower": float(ci_lower_m),
            "ci_upper": float(ci_upper_m),
            "note": "Change in eval (points) per 1 SD increase in beauty for male instructors (beauty + interaction)"
        }

    # Assemble result object
    result_object = {
        "coefficients": {
            "beauty_z": beauty,
            "beauty_z_sq": beauty_sq,
            "beauty_z_x_gender_male": interaction
        },
        "marginal_effects": {
            "female": marg_female,
            "male": marg_male
        },
        "nobs": int(getattr(res, "nobs", -1)) if hasattr(res, "nobs") else None,
        "df_resid": float(getattr(res, "df_resid", np.nan)) if hasattr(res, "df_resid") else None
    }

    # Short textual description to help answer the yes/no question
    # Compose an interpretation based on coefficient signs and significance (alpha=0.05)
    desc_lines = []
    if beauty is None:
        desc_lines.append("Model does not contain 'beauty_z' — cannot evaluate impact of beauty.")
    else:
        b_est = beauty["est"]
        b_p = beauty["p"]
        sign = "positive" if b_est > 0 else ("negative" if b_est < 0 else "null")
        sig = "statistically significant (p < 0.05)" if b_p < 0.05 else "not statistically significant (p >= 0.05)"
        desc_lines.append(f"Main effect (female baseline): beauty_z coefficient = {b_est:.4f}, {sig}; this means a 1 SD increase in perceived attractiveness is associated with a {b_est:.3f}-point change in the evaluation scale (1-5) for female instructors.")
        if interaction is not None:
            i_est = interaction["est"]
            i_p = interaction["p"]
            i_sig = "significant" if i_p < 0.05 else "not significant"
            desc_lines.append(f"The interaction (beauty_z × male) = {i_est:.4f} ({i_sig}); for male instructors the marginal effect is {marg_male['est']:.4f} (p={marg_male['p']:.3f}).")
        else:
            desc_lines.append("No interaction term with gender is present, so the reported beauty effect applies to all instructors as specified.")
        # Note about nonlinearity if beauty_z_sq present
        if beauty_sq is not None:
            desc_lines.append("A quadratic term for beauty is included; interpret the marginal effect with care because the effect of beauty may vary across the beauty scale.")
        # final summary judgement about "impact"
        if (beauty is not None and beauty["p"] < 0.05) or (marg_male is not None and marg_male["p"] < 0.05):
            desc_lines.append("Bottom line: There is evidence that instructor attractiveness affects teaching evaluations (at least for some groups / terms).")
        else:
            desc_lines.append("Bottom line: There is no strong evidence of an effect of attractiveness on teaching evaluations in this model at conventional significance levels.")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}