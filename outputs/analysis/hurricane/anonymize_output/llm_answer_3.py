def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, confidence intervals, and interpretable
    effect sizes (IRR for NB, percent change for OLS on log outcome) for the two
    predictors of interest: 'FemininityIndex_c' and 'FemaleNameBinary'.

    Returns:
      {
        "object": {
          "FemininityIndex_c": {
             "nb_coef": float,
             "nb_se": float,
             "nb_pvalue": float,
             "nb_ci": [lower, upper],
             "nb_IRR": float,
             "nb_IRR_ci": [lower, upper],
             "ols_coef": float,
             "ols_se": float,
             "ols_pvalue": float,
             "ols_ci": [lower, upper],
             "ols_pct_change": float   # 100*(exp(beta)-1)
          },
          "FemaleNameBinary": { ... same fields ... }
        },
        "description": "Human-readable summary of statistical evidence and direction."
      }
    """
    import numpy as np

    # Names of predictors we care about
    predictors = ['FemininityIndex_c', 'FemaleNameBinary']

    # Prepare output structure
    out = {pred: {} for pred in predictors}
    desc_lines = []

    # Helper to safe-extract from results object
    def safe_get(result, attr, default=None):
        try:
            return getattr(result, attr)
        except Exception:
            return default

    # Get negative binomial robust results if available, otherwise nb_model
    nb_res = None
    if isinstance(model_output, dict):
        nb_res = model_output.get('nb_robust') or model_output.get('nb_model')
        ols_res = model_output.get('ols_logfatalities')
    else:
        nb_res = None
        ols_res = None

    # Extract NB stats
    if nb_res is not None:
        try:
            nb_params = np.asarray(nb_res.params)
            nb_index = list(nb_res.params.index)
            nb_bse = np.asarray(nb_res.bse)
            nb_pvalues = np.asarray(nb_res.pvalues)
            nb_ci = np.asarray(nb_res.conf_int())  # shape (k,2)
        except Exception:
            # Try dictionary-like access for older wrappers
            try:
                nb_params = np.asarray(nb_res['params'])
                nb_index = list(nb_res['params'].index)
                nb_bse = np.asarray(nb_res['bse'])
                nb_pvalues = np.asarray(nb_res['pvalues'])
                nb_ci = np.asarray(nb_res['conf_int']())
            except Exception:
                nb_params = nb_index = nb_bse = nb_pvalues = nb_ci = None

        if nb_index is not None:
            for pred in predictors:
                if pred in nb_index:
                    i = nb_index.index(pred)
                    coef = float(nb_params[i])
                    se = float(nb_bse[i]) if nb_bse is not None else None
                    pval = float(nb_pvalues[i]) if nb_pvalues is not None else None
                    ci_low = float(nb_ci[i, 0]) if nb_ci is not None else None
                    ci_high = float(nb_ci[i, 1]) if nb_ci is not None else None
                    irr = float(np.exp(coef))
                    irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                    irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

                    out[pred].update({
                        'nb_coef': coef,
                        'nb_se': se,
                        'nb_pvalue': pval,
                        'nb_ci': [ci_low, ci_high],
                        'nb_IRR': irr,
                        'nb_IRR_ci': [irr_ci_low, irr_ci_high]
                    })
                else:
                    out[pred].update({
                        'nb_coef': None,
                        'nb_se': None,
                        'nb_pvalue': None,
                        'nb_ci': [None, None],
                        'nb_IRR': None,
                        'nb_IRR_ci': [None, None]
                    })
    else:
        # No NB results found
        for pred in predictors:
            out[pred].update({
                'nb_coef': None,
                'nb_se': None,
                'nb_pvalue': None,
                'nb_ci': [None, None],
                'nb_IRR': None,
                'nb_IRR_ci': [None, None]
            })
        desc_lines.append("No Negative Binomial results available in model_output.")

    # Extract OLS on log(1+Fatalities) stats
    if ols_res is not None:
        try:
            ols_params = np.asarray(ols_res.params)
            ols_index = list(ols_res.params.index)
            ols_bse = np.asarray(ols_res.bse)
            ols_pvalues = np.asarray(ols_res.pvalues)
            ols_ci = np.asarray(ols_res.conf_int())
        except Exception:
            try:
                ols_params = np.asarray(ols_res['params'])
                ols_index = list(ols_res['params'].index)
                ols_bse = np.asarray(ols_res['bse'])
                ols_pvalues = np.asarray(ols_res['pvalues'])
                ols_ci = np.asarray(ols_res['conf_int']())
            except Exception:
                ols_params = ols_index = ols_bse = ols_pvalues = ols_ci = None

        if ols_index is not None:
            for pred in predictors:
                if pred in ols_index:
                    j = ols_index.index(pred)
                    coef = float(ols_params[j])
                    se = float(ols_bse[j]) if ols_bse is not None else None
                    pval = float(ols_pvalues[j]) if ols_pvalues is not None else None
                    ci_low = float(ols_ci[j, 0]) if ols_ci is not None else None
                    ci_high = float(ols_ci[j, 1]) if ols_ci is not None else None
                    # Convert coefficient on log(1+y) to percent change approx:
                    pct_change = float((np.exp(coef) - 1) * 100) if coef is not None else None

                    out[pred].update({
                        'ols_coef': coef,
                        'ols_se': se,
                        'ols_pvalue': pval,
                        'ols_ci': [ci_low, ci_high],
                        'ols_pct_change': pct_change
                    })
                else:
                    out[pred].update({
                        'ols_coef': None,
                        'ols_se': None,
                        'ols_pvalue': None,
                        'ols_ci': [None, None],
                        'ols_pct_change': None
                    })
    else:
        for pred in predictors:
            out[pred].update({
                'ols_coef': None,
                'ols_se': None,
                'ols_pvalue': None,
                'ols_ci': [None, None],
                'ols_pct_change': None
            })
        desc_lines.append("No OLS (log fatalities) results available in model_output.")

    # Build human-readable description interpreting NB primarily, with OLS robustness
    for pred in predictors:
        nb_p = out[pred].get('nb_pvalue')
        nb_coef = out[pred].get('nb_coef')
        irr = out[pred].get('nb_IRR')

        # Interpret NB if available
        if nb_p is None:
            desc_lines.append(f"{pred}: no NB estimate available.")
        else:
            sig = nb_p < 0.05
            if nb_coef is None:
                desc_lines.append(f"{pred}: NB estimate missing.")
            else:
                direction = "decrease" if nb_coef < 0 else "increase" if nb_coef > 0 else "no change"
                sig_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"
                # For FemininityIndex_c, interpret per one-unit increase in centered femininity index.
                if pred == 'FemininityIndex_c':
                    desc_lines.append(
                        f"Negative Binomial: For a one-unit increase in femininity (centered), "
                        f"the NB coefficient = {nb_coef:.3g} ({sig_text}), corresponding to an IRR = {irr:.3g}. "
                        f"IRR < 1 implies a {100*(1-irr):.2f}% expected {direction} in fatalities per unit increase."
                    )
                else:
                    # Female binary: from male(0) to female(1)
                    desc_lines.append(
                        f"Negative Binomial: Being female-named (vs. male-named) has coefficient = {nb_coef:.3g} ({sig_text}), "
                        f"IRR = {irr:.3g}. IRR < 1 implies fewer expected fatalities for female-named storms; IRR > 1 implies more."
                    )

        # Add OLS robustness summary if present
        ols_p = out[pred].get('ols_pvalue')
        ols_coef = out[pred].get('ols_coef')
        ols_pct = out[pred].get('ols_pct_change')
        if ols_p is None:
            desc_lines.append(f"{pred}: no OLS(log) estimate available for robustness.")
        else:
            sig2 = ols_p < 0.05
            sig_text2 = "statistically significant (p < 0.05)" if sig2 else "not statistically significant (p >= 0.05)"
            desc_lines.append(
                f"OLS (log1p) robustness: coef = {ols_coef:.3g} ({sig_text2}); "
                f"approx. percent change = {ols_pct:.2f}% for a one-unit increase (or moving to female name for binary)."
            )

    # Synthesize overall take-away (simple rule based on NB significance and sign for femininity variable)
    fem_p = out['FemininityIndex_c'].get('nb_pvalue')
    fem_coef = out['FemininityIndex_c'].get('nb_coef')
    if fem_p is None:
        overall = "No Negative Binomial result for FemininityIndex_c to judge evidence."
    else:
        if fem_p < 0.05:
            if fem_coef < 0:
                overall = "Evidence: Higher perceived femininity of a storm name is associated with significantly fewer fatalities (NB model)."
            elif fem_coef > 0:
                overall = "Evidence: Higher perceived femininity of a storm name is associated with significantly more fatalities (NB model)."
            else:
                overall = "No directional effect (coef ~ 0) despite statistical significance."
        else:
            overall = "No strong evidence in the NB model that femininity of name predicts fatalities (p >= 0.05)."

    desc_lines.append("")  # blank line
    desc_lines.append("Overall conclusion: " + overall)

    description = " ".join(desc_lines)

    return {"object": out, "description": description}