def extract_final_answer(model_output):
    """
    Extracts statistics for the HasChildren coefficient from a fitted statsmodels OLS result
    (with robust SEs, as produced by the modeling function). Returns a dictionary with:
      - "object": a dict of numeric results for the HasChildren effect
      - "description": a short plain-language interpretation answering whether having children
                       decreases engagement in extramarital affairs (based on sign and p-value)
    """
    result = {"object": None, "description": ""}

    # Ensure model_output has the expected attributes
    try:
        params = model_output.params            # pandas Series of coefficients
        bse = model_output.bse                  # standard errors (should reflect cov_type used in fit)
        tvalues = model_output.tvalues
        pvalues = model_output.pvalues
        conf = model_output.conf_int()          # DataFrame/ndarray with 2 columns: lower, upper
    except Exception as e:
        result["description"] = f"Provided model_output does not look like a fitted statsmodels result: {e}"
        return result

    varname = "HasChildren"
    if varname not in params.index:
        result["description"] = f"Variable '{varname}' is not present in the fitted model."
        return result

    coef = float(params[varname])
    se = float(bse[varname]) if varname in bse.index else None
    t = float(tvalues[varname]) if varname in tvalues.index else None
    p = float(pvalues[varname]) if varname in pvalues.index else None

    # Confidence interval: conf_int returns rows indexed by variable names in many versions
    try:
        if hasattr(conf, "loc"):
            ci_lower, ci_upper = float(conf.loc[varname, 0]), float(conf.loc[varname, 1])
        else:
            # assume numpy array and we can find position
            idx = list(params.index).index(varname)
            ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])
    except Exception:
        # fallback: compute using coef +/- 1.96*se if se available
        if se is not None:
            ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se
        else:
            ci_lower = ci_upper = None

    # Decide statistical significance using two-sided p-value (alpha = 0.05)
    significant = (p is not None) and (p < 0.05)

    # Build object to return
    effect_object = {
        "variable": varname,
        "coefficient": coef,
        "std_error": se,
        "t_value": t,
        "p_value": p,
        "conf_int_95_lower": ci_lower,
        "conf_int_95_upper": ci_upper,
        "significant_at_0.05": significant,
        "note_on_scale": (
            "AffairFreq uses the survey's numeric coding (0=none, 1=once, 2=twice, 3=three times, "
            "7=4-10 times, 12=monthly/weekly/daily aggregated). Coefficient is change in that scale "
            "associated with having children (binary indicator)."
        )
    }

    # Formulate interpretation / conclusion
    if p is None:
        conclusion = ("Could not extract a p-value for HasChildren; raw estimate is provided but "
                      "statistical significance cannot be assessed.")
    else:
        if significant:
            if coef < 0:
                conclusion = (
                    f"Having children is associated with a statistically significant decrease in reported "
                    f"affair frequency: coefficient = {coef:.3f} (SE = {se:.3f}, p = {p:.3f}). "
                    f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. Interpretation: married respondents with children "
                    f"report, on average, {abs(coef):.3f} fewer units on the AffairFreq scale than those without children."
                )
            else:
                conclusion = (
                    f"Having children is associated with a statistically significant increase in reported "
                    f"affair frequency: coefficient = {coef:.3f} (SE = {se:.3f}, p = {p:.3f}). "
                    f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]."
                )
        else:
            # Not statistically significant
            if coef < 0:
                conclusion = (
                    f"Point estimate suggests that having children is associated with a decrease in reported affair frequency "
                    f"(coefficient = {coef:.3f}), but this effect is not statistically significant "
                    f"(SE = {se:.3f}, p = {p:.3f}). 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
                    f"Conclusion: there is insufficient evidence to claim having children decreases extramarital affairs."
                )
            else:
                conclusion = (
                    f"Point estimate suggests that having children is associated with an increase in reported affair frequency "
                    f"(coefficient = {coef:.3f}), but this effect is not statistically significant "
                    f"(SE = {se:.3f}, p = {p:.3f}). 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
                    f"Conclusion: there is insufficient evidence of an effect of having children on affair frequency."
                )

    result["object"] = effect_object
    result["description"] = conclusion
    return result