def extract_final_answer(model_output):
    """
    Extract key statistics from a statsmodels GLMResultsWrapper (Poisson or NegBin) that modeled fish counts
    with offset = log(hours). Returns a dict with numeric results and a plain-language description.

    Returned dictionary structure in "object":
      - model_type: 'Poisson' or 'NegativeBinomial' (or best available name)
      - dispersion: deviance / df_resid as stored on the model_output (may be None)
      - baseline_rate_per_hour: exp(intercept) -> expected fish caught per hour when all predictors = 0
      - predictors: list of dicts, one per coefficient, with:
          - name: coefficient name (e.g. 'livebait', 'group_size', 'const')
          - coef: raw coefficient on log-rate scale
          - pvalue: p-value for test coef = 0
          - ci_low, ci_high: 95% CI on the log-rate (coef) scale
          - irr: incidence rate ratio = exp(coef) (multiplicative effect on fish-per-hour)
          - irr_ci_low, irr_ci_high: 95% CI for the IRR (exp(ci_low), exp(ci_high))
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Safely get params, pvalues and conf_int; handle different return types
    try:
        params = res.params.copy()
        pvalues = res.pvalues.copy()
        conf = res.conf_int()
        # conf may be ndarray or DataFrame; convert to DataFrame for easy indexing
        if not isinstance(conf, (pd.DataFrame, pd.Series)):
            conf = pd.DataFrame(conf, index=params.index, columns=['lower', 'upper'])
        else:
            # conf is DataFrame with two columns; ensure names
            if conf.shape[1] >= 2:
                conf = conf.copy()
                conf.columns = ['lower', 'upper']
    except Exception as e:
        raise ValueError("Unable to read parameters/p-values/conf-int from model_output: %s" % str(e))

    # Model metadata
    model_type = getattr(res, 'model_type', None)
    if model_type is None:
        # Try to infer a sensible name
        try:
            model_type = type(res.model.family).__name__
        except Exception:
            model_type = 'Unknown'
    dispersion = getattr(res, 'dispersion', None)

    # Identify intercept name (common names: 'const', 'Intercept')
    intercept_name = None
    for cand in ['const', 'Intercept', 'intercept']:
        if cand in params.index:
            intercept_name = cand
            break
    if intercept_name is None:
        # fallback to the first parameter (may not be ideal but gives a baseline)
        intercept_name = params.index[0]

    # Compute baseline rate per hour = exp(intercept)
    intercept_coef = float(params.loc[intercept_name])
    baseline_rate_per_hour = float(np.exp(intercept_coef))

    # Build predictor table including intercept
    predictors = []
    for name in params.index:
        coef = float(params.loc[name])
        pval = float(pvalues.loc[name]) if name in pvalues.index else None
        # conf may be DataFrame with index matching params.index
        if name in conf.index:
            ci_low = float(conf.loc[name, 'lower'])
            ci_high = float(conf.loc[name, 'upper'])
        else:
            # fallback: try positional
            try:
                pos = list(params.index).index(name)
                ci_low = float(conf.iloc[pos, 0])
                ci_high = float(conf.iloc[pos, 1])
            except Exception:
                ci_low = None
                ci_high = None

        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        predictors.append({
            'name': str(name),
            'coef': coef,
            'pvalue': pval,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'irr': irr,
            'irr_ci_low': irr_ci_low,
            'irr_ci_high': irr_ci_high
        })

    result_object = {
        'model_type': model_type,
        'dispersion': float(dispersion) if dispersion is not None else None,
        'baseline_rate_per_hour': baseline_rate_per_hour,
        'baseline_intercept_name': intercept_name,
        'predictors': predictors
    }

    # Plain-language description
    description_lines = []
    description_lines.append(
        "This model predicts fish count using a log-link GLM with offset = log(hours), so coefficients are on the "
        "log-rate scale. Exponentiating a coefficient gives the incidence rate ratio (IRR): multiplicative effect on "
        "fish caught per hour for a one-unit increase in the predictor."
    )
    description_lines.append(
        f"Baseline rate per hour (all predictors = 0) = exp({intercept_name}) = {baseline_rate_per_hour:.4g} fish/hour."
    )
    if dispersion is not None:
        description_lines.append(
            f"Reported dispersion = {dispersion:.3g}. Model type used: {model_type}."
        )
        if dispersion > 1.5:
            description_lines.append(
                "Dispersion > 1.5 suggests overdispersion; a Negative Binomial model was used/refitted."
            )
    # Summarize each predictor briefly
    for p in predictors:
        # skip repeating baseline intercept too verbosely
        if p['name'] == intercept_name:
            continue
        desc = (f"{p['name']}: coef={p['coef']:.4g}, p={p['pvalue']:.3g}, "
                f"IRR={p['irr']:.3g} (95% CI [{p['irr_ci_low']:.3g}, {p['irr_ci_high']:.3g}])")
        description_lines.append(desc)

    description = " ".join(description_lines)

    return {"object": result_object, "description": description}