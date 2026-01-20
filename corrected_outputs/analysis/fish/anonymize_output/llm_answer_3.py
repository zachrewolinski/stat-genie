def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted model(s) returned by the modeling function.

    Returns a dictionary with:
      - "object": dict containing chosen_model, dispersion, baseline_rate_per_hour,
                  predicted_rate_avg_per_hour (if data available), and a variables
                  dict with coef, se, pvalue, conf_int, irr and irr_conf_int for each term.
      - "description": a short plain-language interpretation of the main results
                       (which model was used, how to interpret IRRs, and the baseline rate).
    """
    import numpy as np
    import pandas as pd

    # Determine which fitted model to use
    chosen = model_output.get('chosen_model', 'poisson')
    if chosen == 'negative_binomial' and 'negative_binomial_model' in model_output:
        res = model_output['negative_binomial_model']
    elif 'poisson_model' in model_output:
        res = model_output['poisson_model']
    else:
        raise ValueError("No fitted model found in model_output.")

    # Extract basic diagnostics
    dispersion = model_output.get('dispersion', None)

    # Extract coefficient table and CIs
    params = res.params.copy()
    bse = res.bse.copy()
    pvalues = res.pvalues.copy()
    try:
        conf = res.conf_int()
        # conf is typically a DataFrame with two columns [0,1]
    except Exception:
        # If conf_int fails, build approximate CIs using normal approximation
        z = 1.96
        conf = pd.DataFrame({
            0: params - z * bse,
            1: params + z * bse
        }, index=params.index)

    variables = {}
    for var in params.index:
        coef = float(params[var])
        se = float(bse[var]) if var in bse.index else None
        pval = float(pvalues[var]) if var in pvalues.index else None
        ci_low = float(conf.loc[var, 0])
        ci_high = float(conf.loc[var, 1])
        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low))
        irr_ci_high = float(np.exp(ci_high))

        variables[var] = {
            'coef': coef,
            'std_error': se,
            'p_value': pval,
            'conf_int': [ci_low, ci_high],
            'incidence_rate_ratio': irr,
            'irr_conf_int': [irr_ci_low, irr_ci_high],
            'interpretation': (
                "Multiplicative effect on fish-per-hour rate. "
                "IRR > 1 -> higher rate; IRR < 1 -> lower rate."
            )
        }

    # Identify intercept name (common names: 'Intercept' or 'const')
    intercept_name = None
    for candidate in ['Intercept', 'const', 'intercept', 'CONST']:
        if candidate in params.index:
            intercept_name = candidate
            break
    # If not found, assume the first parameter is the intercept (should rarely happen)
    if intercept_name is None:
        intercept_name = params.index[0]

    baseline_rate_per_hour = float(np.exp(params[intercept_name]))

    # If underlying data are available in the model, compute predicted rate for average observed covariates
    predicted_rate_avg_per_hour = None
    try:
        # statsmodels keeps the original data in res.model.data.frame when formula API was used
        df = getattr(res.model.data, 'frame', None)
        if df is None:
            # Older/newer versions might store as .frame directly
            df = getattr(res.model.data, 'frame', None)
        if df is None and hasattr(res.model.data, 'orig_exog'):
            # fallback: attempt to reconstruct from exog and exog_names
            exog = getattr(res.model.data, 'orig_exog', None)
            exog_names = getattr(res.model.data, 'orig_exog_names', None)
            if exog is not None and exog_names is not None:
                df = pd.DataFrame(exog, columns=exog_names)
        if df is not None:
            # Use the names provided in the analysis if available; else use all exog columns excluding intercept
            covariates = ['LiveBait', 'Camper', 'Adults', 'Children']
            available_covs = [c for c in covariates if c in df.columns]
            if available_covs:
                means = {c: float(df[c].mean()) for c in available_covs}
                linear = params[intercept_name]
                for c in available_covs:
                    if c in params.index:
                        linear = linear + params[c] * means[c]
                predicted_rate_avg_per_hour = float(np.exp(linear))
    except Exception:
        predicted_rate_avg_per_hour = None

    result_object = {
        'chosen_model': chosen,
        'dispersion': dispersion,
        'baseline_rate_per_hour (all covariates = 0)': baseline_rate_per_hour,
        'predicted_rate_per_hour_for_average_group': predicted_rate_avg_per_hour,
        'variables': variables
    }

    # Build a short description (plain language)
    desc_lines = []
    desc_lines.append(f"Chosen model: {chosen} (dispersion = {dispersion}).")
    desc_lines.append("Reported coefficients are on the log-rate scale; exponentiated coefficients (IRRs) are multiplicative effects on fish-per-hour.")
    desc_lines.append(f"Baseline rate (all predictors = 0) = {baseline_rate_per_hour:.3f} fish per hour.")
    if predicted_rate_avg_per_hour is not None:
        desc_lines.append(f"Predicted rate for an average observed group (means of covariates) = {predicted_rate_avg_per_hour:.3f} fish per hour.")
    desc_lines.append("For each variable, see 'variables' in the returned object: coef, std_error, p_value, 95% CI, and incidence_rate_ratio (IRR) with its 95% CI.")
    description = " ".join(desc_lines)

    return {
        "object": result_object,
        "description": description
    }