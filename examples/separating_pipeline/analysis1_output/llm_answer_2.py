def extract_final_answer(model_output):
    """
    Extracts statistics for the 'masfem_z' coefficient from a fitted statsmodels results object
    (e.g., a GLMResultsWrapper or a robust-covariance results wrapper).

    Returns a dict with:
      - "object": a dict of numeric results (coef, se, t/z, p-value, 95% CI, IRR and IRR CI, percent change)
      - "description": a short text interpretation about whether the result supports the hypothesis
    """
    import math
    # Ensure model_output has needed attributes
    res = model_output

    # Name of variable of interest
    var = 'masfem_z'

    # Helper to raise if something missing
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output has no 'params' attribute. Expected a statsmodels results object.")
    params = res.params

    if var not in params.index:
        raise ValueError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract coefficient
    coef = float(params[var])

    # Standard error (if available)
    se = float(res.bse[var]) if hasattr(res, 'bse') and var in res.bse.index else None

    # t/z value (if available)
    tval = float(res.tvalues[var]) if hasattr(res, 'tvalues') and var in res.tvalues.index else None

    # p-value (if available)
    pval = float(res.pvalues[var]) if hasattr(res, 'pvalues') and var in res.pvalues.index else None

    # 95% confidence interval for coefficient
    try:
        ci_df = res.conf_int()
        if var in ci_df.index:
            ci_low = float(ci_df.loc[var].iloc[0])
            ci_high = float(ci_df.loc[var].iloc[1])
        else:
            # fallback if index not aligned
            ci_low, ci_high = float(ci_df.iloc[params.index.get_loc(var), 0]), float(ci_df.iloc[params.index.get_loc(var), 1])
    except Exception:
        ci_low, ci_high = None, None

    # For a count model with log link, exponentiate coefficient to get incidence rate ratio (IRR)
    irr = math.exp(coef) if coef is not None else None
    irr_ci_low = math.exp(ci_low) if ci_low is not None else None
    irr_ci_high = math.exp(ci_high) if ci_high is not None else None

    # Percent change interpretation
    pct_change = (irr - 1) * 100 if irr is not None else None

    # Build numeric object to return
    result_object = {
        'variable': var,
        'coef': coef,
        'se': se,
        't_or_z': tval,
        'p_value': pval,
        'conf_int_coef_95pct': [ci_low, ci_high],
        'IRR': irr,
        'conf_int_IRR_95pct': [irr_ci_low, irr_ci_high],
        'percent_change_in_expected_fatalities_per_1SD_increase_masfem': pct_change
    }

    # Interpretation relative to the hypothesis:
    # Hypothesis: More feminine names -> perceived as less threatening -> fewer precautions -> more fatalities.
    # That implies a positive coefficient (coef > 0) would support the hypothesis.
    conclusion = ""
    if pval is not None:
        alpha = 0.05
        if pval < alpha:
            if coef > 0:
                conclusion = ("Statistically significant positive association: a one standard-deviation increase in "
                              "name femininity (masfem_z) is associated with higher expected fatalities (p < 0.05). "
                              "This result is consistent with the hypothesis that more feminine names lead to fewer precautions "
                              "and therefore more fatalities.")
            elif coef < 0:
                conclusion = ("Statistically significant negative association: a one standard-deviation increase in "
                              "name femininity is associated with lower expected fatalities (p < 0.05). "
                              "This contradicts the hypothesized direction.")
            else:
                conclusion = ("Coefficient is essentially zero and statistically significant (unexpected).")
        else:
            conclusion = ("No statistically significant association detected (p >= 0.05). "
                          "The data do not provide strong evidence for or against the hypothesis.")
    else:
        conclusion = ("Could not determine statistical significance because p-value unavailable. "
                      "See numeric estimates and confidence intervals for assessment.")

    # Compose human-readable description
    desc_lines = []
    desc_lines.append(f"Coefficient (log scale) for '{var}': {coef:.4f}")
    if se is not None:
        desc_lines.append(f"Standard error: {se:.4f}")
    if tval is not None:
        desc_lines.append(f"t/z-value: {tval:.3f}")
    if pval is not None:
        desc_lines.append(f"p-value: {pval:.3f}")
    if ci_low is not None and ci_high is not None:
        desc_lines.append(f"95% CI for coef: [{ci_low:.4f}, {ci_high:.4f}]")
    if irr is not None:
        desc_lines.append(f"Incidence Rate Ratio (IRR) = exp(coef): {irr:.3f}")
    if irr_ci_low is not None and irr_ci_high is not None:
        desc_lines.append(f"95% CI for IRR: [{irr_ci_low:.3f}, {irr_ci_high:.3f}]")
    if pct_change is not None:
        desc_lines.append(f"Interpreted as: {pct_change:+.1f}% change in expected fatalities per 1 SD increase in femininity.")

    desc_lines.append(conclusion)

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}