def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors of interest
    (age, Sex_M, HelpReceived) from a fitted statsmodels MixedLMResults (or wrapper).
    
    Returns:
      dict with keys:
        - "object": dict mapping each predictor to its extracted statistics:
            { "coef": float,
              "se": float,
              "pvalue": float,
              "ci_lower": float,
              "ci_upper": float,
              "significant": bool (p < 0.05) }
        - "description": short interpretation of what each coefficient means in context.
    """
    # Required predictors to report
    predictors = ['age', 'Sex_M', 'HelpReceived']
    
    # Basic checks
    if not hasattr(model_output, 'params'):
        raise ValueError("model_output does not look like a fitted statsmodels results object (missing .params).")
    
    params = model_output.params
    # bse, pvalues, and conf_int may be attributes or methods
    if hasattr(model_output, 'bse'):
        bse = model_output.bse
    else:
        raise ValueError("model_output missing .bse (standard errors).")
    if hasattr(model_output, 'pvalues'):
        pvalues = model_output.pvalues
    else:
        raise ValueError("model_output missing .pvalues.")
    # conf_int can be a method or attribute
    try:
        conf = model_output.conf_int()
    except TypeError:
        # if conf_int requires no args but raises TypeError, re-call without args
        conf = model_output.conf_int
    except Exception:
        raise ValueError("Could not obtain confidence intervals from model_output via .conf_int().")
    
    result_obj = {}
    for pred in predictors:
        if pred not in params.index:
            # If predictor not found, record None and continue
            result_obj[pred] = {
                "coef": None,
                "se": None,
                "pvalue": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant": None,
                "note": f"{pred} not found in model parameters"
            }
            continue
        # extract values and convert to native Python floats
        coef = float(params.loc[pred])
        se = float(bse.loc[pred])
        pval = float(pvalues.loc[pred]) if pred in pvalues.index else None
        # conf could be a DataFrame-like with rows indexed by param names
        try:
            ci_row = conf.loc[pred]
            ci_lower = float(ci_row.iloc[0])
            ci_upper = float(ci_row.iloc[1])
        except Exception:
            # fallback: try using params +/- 1.96*se
            ci_lower = float(coef - 1.96 * se)
            ci_upper = float(coef + 1.96 * se)
        
        significant = None
        if pval is not None:
            significant = (pval < 0.05)
        
        result_obj[pred] = {
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant
        }
    
    # Short interpretation in the context of nuts/sec efficiency
    desc_lines = [
        "Extracted fixed-effect estimates from the mixed-effects model predicting nut-cracking efficiency (nuts opened per second).",
        "Interpretation for each predictor:",
        "- age: change in efficiency (nuts/sec) per additional year of age.",
        "- Sex_M: estimated difference in efficiency (nuts/sec) for males (Sex_M=1) versus females (Sex_M=0).",
        "- HelpReceived: estimated difference in efficiency (nuts/sec) when the focal individual received help (1) versus not (0).",
        "Each predictor entry includes coefficient, standard error, two-sided p-value, 95% confidence interval, and a boolean 'significant' indicating p < 0.05."
    ]
    description = " ".join(desc_lines)
    
    return {"object": result_obj, "description": description}