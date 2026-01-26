def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs and plain-language interpretations
    for the predictors of interest (Age_z, Sex_Male, Help_Yes) from a fitted statsmodels model
    object (MixedLMResultsWrapper, RegressionResultsWrapper, or similar).

    Returns:
        dict with keys:
          - "object": dict mapping predictor -> dict of numeric results:
                { "coef": float,
                  "se": float,
                  "pvalue": float or None,
                  "ci_lower": float,
                  "ci_upper": float,
                  "significant": bool }
          - "description": str brief human-readable interpretation of each predictor's effect.
    """
    import numpy as np
    import pandas as pd

    predictors = ['Age_z', 'Sex_Male', 'Help_Yes']

    # Prepare containers
    results = {}
    missing = []

    # Helper functions to safely get items
    def get_attr(obj, name):
        return getattr(obj, name, None)

    # Get params (should be a pandas Series)
    params = None
    try:
        params = model_output.params
    except Exception:
        # try attribute .params() or dict-like
        try:
            params = model_output.__dict__.get('params', None)
        except Exception:
            params = None

    # Get bse
    bse = get_attr(model_output, 'bse')
    # Get pvalues
    pvalues = get_attr(model_output, 'pvalues')
    # Get conf_int method result if available
    conf_int_df = None
    try:
        ci = model_output.conf_int()
        # conf_int may be a numpy array or DataFrame
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            conf_int_df = ci
        else:
            # if ndarray, try to turn into DataFrame with index same as params
            if params is not None:
                conf_int_df = pd.DataFrame(ci, index=params.index)
            else:
                conf_int_df = None
    except Exception:
        conf_int_df = None

    # As a fallback, try to get covariance of params to compute se
    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        cov = None

    # Check that params is present
    if params is None:
        raise ValueError("Could not find parameter estimates in the model_output object.")

    # Ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            params = pd.Series(list(params))

    for pred in predictors:
        if pred not in params.index:
            missing.append(pred)
            continue
        coef = float(params.loc[pred])

        # standard error
        se_val = None
        if (isinstance(bse, (pd.Series, dict)) and pred in bse):
            se_val = float(bse[pred])
        elif cov is not None:
            try:
                se_val = float(np.sqrt(float(cov.loc[pred, pred])))
            except Exception:
                se_val = None
        else:
            se_val = None

        # p-value
        pval = None
        if isinstance(pvalues, (pd.Series, dict)) and pred in pvalues:
            try:
                pval = float(pvalues[pred])
            except Exception:
                pval = None

        # confidence interval
        ci_lower = None
        ci_upper = None
        if conf_int_df is not None and pred in conf_int_df.index:
            try:
                row = conf_int_df.loc[pred]
                # row may have two columns [0,1] or named
                if len(row) >= 2:
                    ci_lower = float(row.iloc[0])
                    ci_upper = float(row.iloc[1])
            except Exception:
                ci_lower = None
                ci_upper = None
        elif se_val is not None:
            # approximate 95% CI using normal quantile
            ci_lower = coef - 1.96 * se_val
            ci_upper = coef + 1.96 * se_val

        significant = None
        if pval is not None:
            significant = bool(pval < 0.05)

        results[pred] = {
            "coef": coef,
            "se": se_val,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant
        }

    # Build a short human-readable description
    descr_lines = []
    model_name = type(model_output).__name__
    descr_lines.append(f"Model object type: {model_name}. Extracted estimates for predictors: {', '.join([p for p in predictors if p in params.index])}.")

    for pred in predictors:
        if pred not in results:
            descr_lines.append(f"- {pred}: NOT included in the fitted model.")
            continue
        r = results[pred]
        coef = r["coef"]
        se_val = r["se"]
        pval = r["pvalue"]
        ci_l = r["ci_lower"]
        ci_u = r["ci_upper"]
        sig = r["significant"]

        # Format numbers succinctly
        def fmt(x):
            if x is None:
                return "NA"
            return f"{x:.4f}"

        direction = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        sig_text = "statistically significant (p < 0.05)" if sig else ("not statistically significant (p >= 0.05)" if sig is not None else "p-value unavailable")

        interpretation = ""
        if pred == 'Age_z':
            interpretation = f"Per 1 SD increase in age, efficiency changes by {fmt(coef)} nuts/sec."
        elif pred == 'Sex_Male':
            interpretation = f"Male chimpanzees differ from females by {fmt(coef)} nuts/sec (male minus female)."
        elif pred == 'Help_Yes':
            interpretation = f"Receiving help (yes vs no) changes efficiency by {fmt(coef)} nuts/sec."

        descr_lines.append(
            f"- {pred}: coef={fmt(coef)}, se={fmt(se_val)}, 95% CI=[{fmt(ci_l)}, {fmt(ci_u)}], p={fmt(pval)} -> {direction} effect; {sig_text}. {interpretation}"
        )

    if missing:
        descr_lines.append("Note: Some predictors were not present in the model: " + ", ".join(missing) + ".")

    description = " ".join(descr_lines)

    return {"object": results, "description": description}