def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, 95% CIs, and incidence-rate-ratios (IRRs)
    for the predictors of interest (age, sex_male, help_received) from a statsmodels
    GLMResultsWrapper (regular or robustcov results).
    Returns:
      {
        "object": { varname: { 'coef': ..., 'se': ..., 'pvalue': ..., 'ci95': (...,...),
                               'irr': ..., 'irr_ci95': (...,...) , 'unit_interp': ... } , ... },
        "description": "textual interpretation"
      }
    """
    import numpy as np
    import pandas as pd

    # Variables of interest
    vars_of_interest = ['age', 'sex_male', 'help_received']

    out = {}
    # Attempt to access common statsmodels result attributes
    try:
        params = model_output.params        # pandas Series
        bse = model_output.bse              # pandas Series
        pvalues = model_output.pvalues      # pandas Series
        ci = model_output.conf_int()        # DataFrame with two columns [lower, upper]
    except Exception as e:
        raise ValueError("model_output does not have expected statsmodels result attributes.") from e

    for v in vars_of_interest:
        if v not in params.index:
            out[v] = {
                'present': False,
                'message': f"Variable '{v}' not found in the model coefficients."
            }
            continue

        coef = float(params.loc[v])
        se = float(bse.loc[v]) if v in bse.index else None
        p = float(pvalues.loc[v]) if v in pvalues.index else None
        ci_lower, ci_upper = (None, None)
        if v in ci.index:
            ci_lower, ci_upper = float(ci.loc[v, 0]), float(ci.loc[v, 1])

        # For a Poisson model with log link and log(seconds) offset, coefficients are log-rate-ratios.
        irr = float(np.exp(coef))
        irr_ci_lower, irr_ci_upper = (None, None)
        if ci_lower is not None and ci_upper is not None:
            irr_ci_lower, irr_ci_upper = float(np.exp(ci_lower)), float(np.exp(ci_upper))

        # Human-readable interpretation of one-unit change
        if v == 'age':
            unit_interp = (f"Each additional year of age is associated with a multiplicative change "
                           f"of {irr:.3f} in the nut-opening rate (nuts per second).")
        elif v == 'sex_male':
            unit_interp = (f"Being male (vs female) is associated with a multiplicative change "
                           f"of {irr:.3f} in the nut-opening rate (nuts per second).")
        elif v == 'help_received':
            unit_interp = (f"Receiving help (vs no help) during the session is associated with a multiplicative change "
                           f"of {irr:.3f} in the nut-opening rate (nuts per second).")
        else:
            unit_interp = ""

        out[v] = {
            'present': True,
            'coef': coef,
            'se': se,
            'pvalue': p,
            'ci95_coef': (ci_lower, ci_upper),
            'irr': irr,
            'ci95_irr': (irr_ci_lower, irr_ci_upper),
            'unit_interpretation': unit_interp
        }

    # Summarize significance decisions at alpha = 0.05
    decisions = []
    for v in vars_of_interest:
        info = out.get(v, {})
        if not info.get('present', False):
            decisions.append(f"{v}: not in model.")
            continue
        p = info['pvalue']
        if p is None:
            decisions.append(f"{v}: p-value unavailable.")
        else:
            sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p >= 0.05)"
            decisions.append(f"{v}: {sig}; coefficient = {info['coef']:.4f}; IRR = {info['irr']:.3f}; p = {p:.3g}")

    description = (
        "Extracted results for predictors of nut-cracking rate (Poisson GLM with log(seconds) offset).\n"
        "For each predictor, 'coef' is the log-rate-ratio, 'irr' = exp(coef) is the multiplicative effect on the "
        "nuts-per-second rate, 'ci95_coef' is the 95% CI on the log scale, and 'ci95_irr' is the 95% CI for the IRR.\n\n"
        "Significance summary (alpha = 0.05):\n- " + "\n- ".join(decisions) + "\n\n"
        "Interpretation notes:\n"
        "- For 'age': IRR > 1 means older individuals open nuts faster (higher rate) per additional year; IRR < 1 means slower.\n"
        "- For 'sex_male': IRR compares males to females (male / female). IRR > 1 means males have higher rate.\n"
        "- For 'help_received': IRR compares sessions with help to sessions without help. IRR > 1 means help is associated with higher rate.\n\n"
        "Return object 'object' contains numeric results per variable for further programmatic use."
    )

    return {"object": out, "description": description}