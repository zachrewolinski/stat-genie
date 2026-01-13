def extract_final_answer(model_output):
    """
    Extracts key statistics from the model output of a Negative Binomial GLM
    that modeled fish counts with offset = log(hours).
    Returns a dict with:
      - "object": detailed numeric results (coefficients, p-values, CIs, rate ratios, baseline rate, dispersion)
      - "description": plain-language interpretation of the main quantities
    
    Expected input: model_output is the dict returned by the provided model()
    (keys: 'results', 'dispersion_approx', 'exp_coef', 'model_formula_predictors')
    """
    import numpy as np
    import pandas as pd

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model function.")

    results = model_output.get('results')
    if results is None:
        raise ValueError("model_output missing 'results' key.")

    # Extract parameter estimates and diagnostics
    params = results.params          # coefficients on log-rate scale
    bse = getattr(results, 'bse', None)
    pvalues = getattr(results, 'pvalues', None)
    try:
        conf = results.conf_int()    # default 95% CI on coefficient scale
        conf_df = pd.DataFrame(conf, index=params.index, columns=['2.5%', '97.5%'])
    except Exception:
        conf_df = pd.DataFrame(index=params.index, columns=['2.5%', '97.5%'])

    # Exponentiated coefficients = multiplicative effect on fish-per-hour (rate ratios)
    exp_params = np.exp(params)
    exp_conf_df = np.exp(conf_df.astype(float))

    # Baseline rate per hour: exp(const) (since model is E[fish] = hours * exp(X beta))
    baseline_rate = None
    if 'const' in params.index:
        baseline_rate = float(np.exp(params['const']))
    elif params.shape[0] > 0:
        # fallback: first parameter interpreted as intercept if named differently
        baseline_rate = float(np.exp(params.iloc[0]))

    # Package numeric outputs into JSON-serializable dicts
    def series_to_dict(s):
        return {str(k): (float(v) if (np.ndim(v) == 0 and not pd.isna(v)) else v) for k, v in s.items()}

    object_dict = {
        'coefficients_log_rate': series_to_dict(params),
        'std_errors': series_to_dict(bse) if bse is not None else None,
        'p_values': series_to_dict(pvalues) if pvalues is not None else None,
        'conf_int_log_rate': {str(k): [float(v1), float(v2)] for k, (v1, v2) in conf_df.iterrows()},
        'rate_ratios_per_hour': series_to_dict(exp_params),
        'rate_ratio_conf_int_per_hour': {str(k): [float(v1), float(v2)] for k, (v1, v2) in exp_conf_df.iterrows()},
        'baseline_rate_per_hour': baseline_rate,  # fish per hour when predictors = 0
        'dispersion_approx': float(model_output.get('dispersion_approx', np.nan)),
        'model_predictors': model_output.get('model_formula_predictors', [])
    }

    # Short interpretation string
    description_lines = []
    description_lines.append(
        "This model uses an offset = log(hours), so coefficients model the log(rate) "
        "of fish caught per hour. Exponentiated coefficients are multiplicative effects "
        "on the fish-per-hour rate."
    )
    if baseline_rate is not None:
        description_lines.append(
            f"The baseline rate (all predictors = 0) is approximately {baseline_rate:.3g} fish/hour "
            "(this is exp(intercept))."
        )
    description_lines.append(
        "Rate ratios > 1 indicate a higher catch rate per hour when that predictor increases by 1 unit; "
        "rate ratios < 1 indicate a reduced catch rate."
    )
    description_lines.append(
        "The returned fields include: log-scale coefficients, standard errors, p-values, 95% CIs, "
        "exponentiated coefficients (rate ratios) and their CIs, and a dispersion approximation "
        "(Pearson chi2 / df). A dispersion < 1 (here reported in 'dispersion_approx') suggests "
        "under-dispersion relative to the assumed variance function."
    )

    description = " ".join(description_lines)

    return {"object": object_dict, "description": description}