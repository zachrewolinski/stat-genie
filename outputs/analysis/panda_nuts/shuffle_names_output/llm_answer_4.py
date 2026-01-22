def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% confidence intervals
    for the predictors age_years, sex_m, and received_help from a statsmodels
    RegressionResultsWrapper (which is expected to have been fit with clustered
    standard errors as in the modeling code).

    Returns:
      {
        "object": { var_name: { "coef": ..., "se": ..., "p": ..., "ci_lower": ..., "ci_upper": ...,
                                "pct_change": ..., "ci_pct_lower": ..., "ci_pct_upper": ..., "significant": True/False },
                    ... },
        "description": "text summary explaining the meaning and statistical significance of each effect"
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output
    vars_of_interest = ['age_years', 'sex_m', 'received_help']

    # Prepare containers
    out = {}
    lines = []
    # Try to get standard result arrays; handle missing gracefully
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int()  # should reflect the fitted covariance (clustered if used)
    except Exception as e:
        raise ValueError(f"Unable to extract basic statistics from model_output: {e}")

    for v in vars_of_interest:
        if v not in params.index:
            out[v] = None
            lines.append(f"{v}: NOT FOUND in model output.")
            continue

        coef = float(params.loc[v])
        se = float(bse.loc[v]) if v in bse.index else None
        p = float(pvalues.loc[v]) if v in pvalues.index else None
        # conf_int may be a DataFrame with columns [0,1]
        try:
            ci_lower = float(conf.loc[v][0])
            ci_upper = float(conf.loc[v][1])
        except Exception:
            # fallback: use coef +/- 1.96*se (large-sample normal approx)
            if se is not None:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = ci_upper = None

        # Because outcome is log(nuts_per_sec), interpret coefficient as approximate percent change:
        try:
            pct_change = (np.exp(coef) - 1.0) * 100.0
            ci_pct_lower = (np.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
            ci_pct_upper = (np.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
        except Exception:
            pct_change = ci_pct_lower = ci_pct_upper = None

        significant = (p is not None) and (p < 0.05)

        out[v] = {
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pct_change": pct_change,
            "ci_pct_lower": ci_pct_lower,
            "ci_pct_upper": ci_pct_upper,
            "significant_at_0.05": bool(significant)
        }

        # Build a human-readable line for this predictor
        # Interpretations:
        if v == 'age_years':
            iv_label = "Age (years)"
            iv_interpret = ("Each additional year of age is associated with "
                            f"{pct_change:.2f}% change in nut-cracking efficiency (nuts/sec) on average.")
        elif v == 'sex_m':
            iv_label = "Sex (male vs female; 1=male)"
            iv_interpret = ("Being male (sex_m=1) is associated with "
                            f"{pct_change:.2f}% change in nut-cracking efficiency compared to female (sex_m=0).")
        else:  # received_help
            iv_label = "Received help (1 = yes)"
            iv_interpret = ("Receiving help (received_help=1) is associated with "
                            f"{pct_change:.2f}% change in nut-cracking efficiency compared to not receiving help.")

        sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p ≥ 0.05)"
        line = (f"{iv_label}: coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4f}, "
                f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. {iv_interpret} "
                f"95% CI for percent change = [{ci_pct_lower:.2f}%, {ci_pct_upper:.2f}%]. "
                f"Effect is {sig_text}.")
        lines.append(line)

    description = ("Summary of effects on log(nuts_per_sec) (model used clustered SEs by chimp_id):\n"
                   + "\n".join(lines)
                   + "\n\nNotes: Because the dependent variable is log(nuts/sec), coefficients are shown as"
                   " approximate percent changes via (exp(coef)-1)*100. sex_m is coded 1 = male, 0 = female;"
                   " received_help is coded 1 = yes, 0 = no.")

    return {"object": out, "description": description}