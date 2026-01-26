def extract_final_answer(model_output):
    """
    Extracts the estimated effect(s) of the centered masculinity-femininity index (MasFem_c)
    from a fitted statsmodels OLS RegressionResultsWrapper that used the formula:
      LogAllDeaths ~ MasFem_c * C(Category) + Wind + MinPressure + Year + ElapsedYears + Gender_MF

    Returns a dictionary with:
      - "object": a dict containing the main MasFem_c coefficient, its SE, p-value, 95% CI,
                  any MasFem_c x Category interaction coefficients, and marginal effects
                  of MasFem_c for the reference category and for each observed Category level.
      - "description": A short interpretation of those statistics in context.

    Notes:
      - Uses the model's robust covariance matrix (model_output.cov_params()) to compute
        standard errors for linear combinations (marginal effects by Category).
      - If interaction terms are present they are expected to contain 'MasFem_c' and 'C(Category)'.
    """
    import numpy as np
    from math import erf, sqrt

    def _p_from_z(z):
        # two-sided p-value from z using the error function (no external scipy dependency)
        return 2 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

    # Validate input object
    if not hasattr(model_output, "params") or not hasattr(model_output, "cov_params"):
        raise ValueError("model_output must be a fitted statsmodels results object with .params and .cov_params()")

    params = model_output.params
    cov = model_output.cov_params()

    # Ensure main MasFem_c is present
    if "MasFem_c" not in params.index:
        raise ValueError("MasFem_c not found in model params index")

    results = {}
    # Main effect
    main_coef = float(params["MasFem_c"])
    main_var = float(cov.loc["MasFem_c", "MasFem_c"])
    main_se = float(np.sqrt(main_var))
    main_z = main_coef / main_se if main_se > 0 else np.nan
    main_p = _p_from_z(main_z) if main_se > 0 else np.nan
    main_ci_lower = main_coef - 1.96 * main_se
    main_ci_upper = main_coef + 1.96 * main_se

    results["main_effect"] = {
        "term": "MasFem_c (baseline/reference Category)",
        "coef": main_coef,
        "se": main_se,
        "z_or_t": main_z,
        "p_value": main_p,
        "95%_CI": (main_ci_lower, main_ci_upper),
        "interpretation_brief": (
            "Change in log(1+all deaths) associated with a one-unit increase in the centered "
            "masculinity-femininity index for the reference (omitted) Category."
        ),
    }

    # Find interaction terms involving MasFem_c and Category
    interaction_terms = [n for n in params.index if ("MasFem_c" in n) and ("C(Category)" in n) and (n != "MasFem_c")]
    interactions = {}
    marginal_effects = {}

    # Reference category marginal effect is the main effect
    marginal_effects["Reference (baseline Category)"] = {
        "coef": main_coef,
        "se": main_se,
        "z_or_t": main_z,
        "p_value": main_p,
        "95%_CI": (main_ci_lower, main_ci_upper),
    }

    # For each interaction, compute interaction coef and marginal effect for that category
    for iterm in interaction_terms:
        # try to extract a readable level label from the parameter name
        # typical param name shapes:
        #  - 'MasFem_c:C(Category)[T.2]' or 'MasFem_c:C(Category)[T.3]'
        #  - or sometimes 'MasFem_c:C(Category)[T.1.0]' etc.
        # We'll extract text inside the trailing brackets if present, else use full name.
        level_label = iterm
        if "[" in iterm and "]" in iterm:
            try:
                inside = iterm.split("[", 1)[1].split("]", 1)[0]
                # inside often like 'T.2' -> convert to '2' or keep as 'T.2' if unexpected
                if inside.startswith("T."):
                    level_label = inside[2:]
                else:
                    level_label = inside
            except Exception:
                level_label = iterm

        int_coef = float(params[iterm])
        int_var = float(cov.loc[iterm, iterm]) if iterm in cov.index else 0.0
        int_se = float(np.sqrt(int_var)) if int_var >= 0 else float("nan")

        # Marginal effect for this category = main_coef + int_coef
        marg_coef = main_coef + int_coef
        # var(marg) = var(main) + var(int) + 2*cov(main, int)
        cov_main_int = float(cov.loc["MasFem_c", iterm]) if (("MasFem_c" in cov.index) and (iterm in cov.index)) else 0.0
        marg_var = main_var + int_var + 2.0 * cov_main_int
        marg_se = float(np.sqrt(marg_var)) if marg_var >= 0 else float("nan")
        marg_z = marg_coef / marg_se if marg_se > 0 else float("nan")
        marg_p = _p_from_z(marg_z) if marg_se > 0 else float("nan")
        marg_ci = (marg_coef - 1.96 * marg_se, marg_coef + 1.96 * marg_se) if marg_se == marg_se else (None, None)

        interactions[level_label] = {
            "interaction_term_name": iterm,
            "coef": int_coef,
            "se": int_se,
            "95%_CI": (int_coef - 1.96 * int_se, int_coef + 1.96 * int_se) if int_se == int_se else (None, None),
            "interpretation_brief": f"Adjustment to the MasFem_c slope when Category == {level_label} (added to baseline MasFem_c).",
        }

        marginal_effects[f"Category == {level_label}"] = {
            "coef": marg_coef,
            "se": marg_se,
            "z_or_t": marg_z,
            "p_value": marg_p,
            "95%_CI": marg_ci,
            "interpretation_brief": (
                f"Estimated effect of MasFem_c on log(1+deaths) specifically for Category == {level_label} "
                "(sum of baseline MasFem_c coef and the MasFem_c x Category interaction)."
            ),
        }

    results["interactions"] = interactions
    results["marginal_effects_by_category"] = marginal_effects

    # Short description interpreting direction relative to the hypothesis:
    # Hypothesis: more feminine names -> perceived less threatening -> fewer precautions -> higher fatalities.
    # So a positive coefficient supports the hypothesis (more feminine -> higher log deaths).
    if np.isnan(results["main_effect"]["p_value"]):
        significance_text = "p-value for the main effect could not be computed."
    else:
        sig = results["main_effect"]["p_value"]
        if sig < 0.001:
            significance_text = "main effect is statistically significant at p < 0.001"
        elif sig < 0.01:
            significance_text = "main effect is statistically significant at p < 0.01"
        elif sig < 0.05:
            significance_text = "main effect is statistically significant at p < 0.05"
        else:
            significance_text = "main effect is not statistically significant (p >= 0.05)"

    direction = "positive (more feminine -> higher log deaths)" if main_coef > 0 else "negative (more feminine -> lower log deaths)" if main_coef < 0 else "null (coef = 0)"

    description = (
        f"Extracted the main MasFem_c coefficient and MasFem_c x Category interactions (if present).\n"
        f"Main MasFem_c effect: coef = {main_coef:.4f}, SE = {main_se:.4f}, p ≈ {main_p:.4g}, 95% CI = ({main_ci_lower:.4f}, {main_ci_upper:.4f}).\n"
        f"Direction: {direction}. By the study hypothesis, a positive coefficient would support the claim that "
        f"more feminine names are associated with higher fatalities (consistent with fewer precautions).\n"
        f"{significance_text}.\n"
        f"Marginal effects by Category are provided under 'object' -> 'marginal_effects_by_category'. "
        f"Each marginal effect equals the baseline MasFem_c slope plus the MasFem_c:C(Category) adjustment; "
        f"SEs and CIs use the model covariance to account for correlation between terms."
    )

    return {"object": results, "description": description}