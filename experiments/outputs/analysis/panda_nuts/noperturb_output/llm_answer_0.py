def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, z-stats, p-values, 95% CIs, and incidence-rate-ratios (IRRs)
    for the predictors of interest (age_z, sex_M, help_Y) from a statsmodels GLMResults-like object.
    
    Returns:
      dict with keys:
        - "object": pandas.DataFrame with rows for each variable and columns
                    ['coef','se','z','p','ci_lower','ci_upper','irr','irr_ci_lower','irr_ci_upper']
        - "description": human-readable summary interpreting each predictor's effect
    """
    import numpy as np
    import pandas as pd

    # Variables of interest
    vars_of_interest = ['age_z', 'sex_M', 'help_Y']

    # Check that model_output looks like a statsmodels results object
    # We'll attempt to read params, bse, pvalues, and conf_int; otherwise try fallbacks.
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("model_output does not expose .params. Provide a statsmodels results object.") from e

    # Build containers
    rows = []
    present_vars = [v for v in vars_of_interest if v in params.index]

    if not present_vars:
        raise ValueError(f"None of the requested variables {vars_of_interest} are present in the model output.")

    # Try to obtain bse, pvalues, conf_int (these should reflect robust cov if model_output was a robust wrapper)
    # Fall back to computing bse from cov_params if necessary.
    try:
        bse_all = model_output.bse
    except Exception:
        try:
            cov = model_output.cov_params()
            bse_all = np.sqrt(np.diag(cov))
            bse_all = pd.Series(bse_all, index=model_output.params.index)
        except Exception:
            bse_all = None

    try:
        pvalues_all = model_output.pvalues
    except Exception:
        pvalues_all = None

    try:
        ci = model_output.conf_int()
        # conf_int returns a DataFrame with two columns (lower, upper)
        ci.columns = ['ci_lower', 'ci_upper']
    except Exception:
        # compute Wald-style 95% CIs using bse if available
        if bse_all is None:
            ci = None
        else:
            z = 1.96
            ci_lower = params - z * bse_all
            ci_upper = params + z * bse_all
            ci = pd.DataFrame({'ci_lower': ci_lower, 'ci_upper': ci_upper})

    for v in present_vars:
        coef = float(params.loc[v])
        se = float(bse_all.loc[v]) if (bse_all is not None and v in getattr(bse_all, 'index', bse_all)) else np.nan
        z_stat = coef / se if (not np.isnan(se) and se != 0) else np.nan
        pval = float(pvalues_all.loc[v]) if (pvalues_all is not None and v in pvalues_all.index) else np.nan
        if ci is not None and v in ci.index:
            ci_low = float(ci.loc[v, 'ci_lower'])
            ci_high = float(ci.loc[v, 'ci_upper'])
        else:
            ci_low = float(coef - 1.96 * se) if (not np.isnan(se)) else np.nan
            ci_high = float(coef + 1.96 * se) if (not np.isnan(se)) else np.nan

        irr = float(np.exp(coef))
        irr_ci_lower = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
        irr_ci_upper = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan

        rows.append({
            'variable': v,
            'coef': coef,
            'se': se,
            'z': z_stat,
            'p': pval,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'irr': irr,
            'irr_ci_lower': irr_ci_lower,
            'irr_ci_upper': irr_ci_upper
        })

    df = pd.DataFrame(rows).set_index('variable')

    # Build a concise human-readable interpretation
    lines = []
    alpha = 0.05
    for v, r in df.iterrows():
        # Rounded values for readability
        coef_r = round(r['coef'], 3) if not np.isnan(r['coef']) else 'NA'
        irr_r = round(r['irr'], 3) if not np.isnan(r['irr']) else 'NA'
        ci_r = (round(r['irr_ci_lower'], 3), round(r['irr_ci_upper'], 3)) if not np.isnan(r['irr_ci_lower']) else ('NA', 'NA')
        p_r = round(r['p'], 3) if not np.isnan(r['p']) else 'NA'

        sig = False
        if not np.isnan(r['p']):
            sig = r['p'] < alpha

        # Interpretation text differs slightly by variable type
        if v == 'age_z':
            var_desc = ("Age (standardized): for a one-SD increase in age, the model-implied "
                        f"nut-opening rate multiplies by {irr_r} (95% CI {ci_r}), p = {p_r}.")
        elif v == 'sex_M':
            var_desc = ("Sex (male vs female): being male (1) compared to female (0) is associated with "
                        f"rate multiplier {irr_r} (95% CI {ci_r}), p = {p_r}.")
        elif v == 'help_Y':
            var_desc = ("Help received (yes vs no): receiving help (1) compared to not (0) is associated with "
                        f"rate multiplier {irr_r} (95% CI {ci_r}), p = {p_r}.")
        else:
            var_desc = (f"{v}: coef={coef_r}, IRR={irr_r}, 95% CI={ci_r}, p={p_r}.")

        sig_text = "Statistically significant at alpha=0.05." if sig else "Not statistically significant at alpha=0.05."
        lines.append(f"{var_desc} {sig_text}")

    description = "Effects on nut-cracking efficiency (rate of nuts opened per second) for predictors:\n" + "\n".join(lines)

    return {"object": df, "description": description}