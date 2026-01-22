def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of player skin tone on red cards
    from the model_output dictionary returned by the modeling function.

    Returns:
      {
        "object": {
           "binary": {
             "param_name": str,
             "coef": float,
             "std_err": float,
             "p_value": float,
             "conf_int": [float_lower, float_upper],
             "IRR": float,                 # incidence rate ratio = exp(coef)
             "IRR_conf_int": [float, float]
           } | None,
           "continuous": { ... } | None,
           "poisson_dispersion_primary": float | None
        },
        "description": str  # short plain-language interpretation answering the
                           # question whether darker-skinned players receive more red cards.
      }
    """

    import numpy as np

    def choose_result(key_pref):
        # Prefer clustered results if present, otherwise raw
        for key in (f'nb_{key_pref}_clustered', f'nb_{key_pref}_raw'):
            if key in model_output and model_output[key] is not None:
                return model_output[key]
        return None

    def find_param_name(res, target):
        """
        Try to find the parameter name in the result object that corresponds to target.
        Matches exact name first, then any name that contains the target substring.
        Returns None if not found.
        """
        try:
            names = list(res.params.index)
        except Exception:
            return None
        if target in names:
            return target
        # look for names containing target (useful if patsy encoded factors)
        for n in names:
            if target in n:
                return n
        return None

    def summarize_param(res, param_name):
        if res is None or param_name is None:
            return None
        try:
            coef = float(res.params[param_name])
        except Exception:
            coef = None
        try:
            se = float(res.bse[param_name])
        except Exception:
            se = None
        try:
            pval = float(res.pvalues[param_name])
        except Exception:
            pval = None
        # confidence interval (95% by default)
        try:
            ci = res.conf_int().loc[param_name].values.astype(float)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower, ci_upper = None, None

        irr = np.exp(coef) if coef is not None else None
        irr_ci = [np.exp(ci_lower) if ci_lower is not None else None,
                  np.exp(ci_upper) if ci_upper is not None else None]

        return {
            "param_name": param_name,
            "coef": coef,
            "std_err": se,
            "p_value": pval,
            "conf_int": [ci_lower, ci_upper],
            "IRR": irr,
            "IRR_conf_int": irr_ci
        }

    # Get results objects
    binary_res = choose_result('binary')
    cont_res = choose_result('continuous')

    # Find param names
    bin_param = None
    if binary_res is not None:
        bin_param = find_param_name(binary_res, 'DarkBinary')

    cont_param = None
    if cont_res is not None:
        cont_param = find_param_name(cont_res, 'SkinTone')

    # Summarize
    binary_summary = summarize_param(binary_res, bin_param)
    continuous_summary = summarize_param(cont_res, cont_param)

    # Poisson dispersion if present
    pdiff = model_output.get('poisson_dispersion_primary', None)

    # Formulate a short interpretation regarding the research question
    def interpret_entry(entry, label):
        if entry is None:
            return f"No {label} estimate available."
        p = entry.get('p_value')
        irr = entry.get('IRR')
        ci = entry.get('IRR_conf_int')
        if p is None:
            sig_text = "p-value unavailable"
        else:
            sig_text = ("statistically significant (p < 0.05)"
                        if p < 0.05 else f"not statistically significant (p = {p:.3f})")
        if irr is None:
            return f"{label}: estimate available but could not compute IRR."
        # Interpret direction
        if ci[0] is not None and ci[1] is not None:
            if ci[0] > 1:
                direction = "a significantly higher rate of red cards for the higher-coded group"
            elif ci[1] < 1:
                direction = "a significantly lower rate of red cards for the higher-coded group"
            else:
                direction = "no statistically significant difference in the rate of red cards"
        else:
            direction = "insufficient CI information to determine significance/direction"
        return (f"{label}: IRR = {irr:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f} if available), "
                f"{sig_text}; interpretation: {direction}.")

    bin_interp = interpret_entry(binary_summary, "Binary Dark vs Light")
    cont_interp = interpret_entry(continuous_summary, "Continuous SkinTone (0-1)")

    # Conclude overall: prefer binary if available, otherwise continuous.
    conclusion = ""
    # Determine significance from binary first
    def conclude_from(entry):
        if entry is None:
            return None
        p = entry.get('p_value')
        irr = entry.get('IRR')
        ci = entry.get('IRR_conf_int')
        if p is None:
            return None
        # Return True if significantly >1
        if p < 0.05 and irr is not None and ci[0] is not None and ci[0] > 1:
            return "positive"
        if p < 0.05 and irr is not None and ci[1] is not None and ci[1] < 1:
            return "negative"
        return "null"

    for entry in (binary_summary, continuous_summary):
        c = conclude_from(entry)
        if c == "positive":
            conclusion = "Overall conclusion: Evidence that darker-skinned players receive more red cards (statistically significant positive association)."
            break
        if c == "negative":
            conclusion = "Overall conclusion: Evidence that darker-skinned players receive fewer red cards (statistically significant negative association)."
            break
    if conclusion == "":
        # No strong significant positive/negative effect
        # If either model has an IRR >1 but not significant, note weak/non-significant tendency
        noted = False
        for entry in (binary_summary, continuous_summary):
            if entry is None:
                continue
            irr = entry.get('IRR')
            p = entry.get('p_value')
            if irr is not None and irr > 1 and (p is None or p >= 0.05):
                conclusion = ("Overall conclusion: No statistically significant evidence that darker-skinned players receive more red cards; "
                              "point estimates (IRR > 1) suggest a possible positive association but it is not statistically significant.")
                noted = True
                break
        if not noted:
            conclusion = ("Overall conclusion: No statistically significant evidence that darker-skinned players receive more red cards "
                          "in the provided models (estimates are null or not reliably different from 1).")

    description_lines = [
        "Extracted estimates and interpretation for the effect of player skin tone on red cards.",
        bin_interp,
        cont_interp,
        f"Poisson dispersion on primary sample (diagnostic): {pdiff}",
        conclusion
    ]
    description = " ".join(description_lines)

    return {
        "object": {
            "binary": binary_summary,
            "continuous": continuous_summary,
            "poisson_dispersion_primary": pdiff
        },
        "description": description
    }