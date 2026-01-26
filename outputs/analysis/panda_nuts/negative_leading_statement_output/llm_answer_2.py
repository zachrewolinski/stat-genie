def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, test statistics, p-values, and 95% CIs
    for the predictors of interest (age, Sex_M, Help) from a statsmodels
    MixedLMResults (or wrapper) object.

    Returns:
      {
        "object": {
          "age": { "coef": ..., "se": ..., "z": ..., "p": ..., "ci_lower": ..., "ci_upper": ..., "exp_coef": ... },
          "Sex_M": { ... },
          "Help": { ... }
        },
        "description": "Short human-readable summary of the effects and interpretation"
      }
    """
    import math
    from collections import OrderedDict

    # Helper to get attribute if exists, else None
    def _getattr(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else None

    # Extract fixed-effect coefficients
    fe_params = _getattr(model_output, 'fe_params')
    if fe_params is None:
        # fallback to params (may include other params too)
        fe_params = _getattr(model_output, 'params')
    # Standard errors for fixed effects
    bse_fe = _getattr(model_output, 'bse_fe')
    if bse_fe is None:
        bse_fe = _getattr(model_output, 'bse')

    # p-values (may exist)
    pvalues = _getattr(model_output, 'pvalues')

    # Confidence intervals method
    conf_int = None
    try:
        # conf_int() returns a DataFrame-like object
        conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    # Ensure we can index fe_params like a dict/Series
    try:
        keys = list(fe_params.index)
    except Exception:
        # convert to dict-like if necessary
        try:
            fe_params = dict(fe_params)
            keys = list(fe_params.keys())
        except Exception:
            raise ValueError("Unable to read fixed-effect parameters from model_output")

    predictors = ['age', 'Sex_M', 'Help']
    results = OrderedDict()
    for pred in predictors:
        if pred not in keys:
            results[pred] = {
                "available": False,
                "message": f"Predictor '{pred}' not found in model fixed effects."
            }
            continue

        coef = float(fe_params[pred])
        se = None
        # get standard error if available
        try:
            if hasattr(bse_fe, 'get') or isinstance(bse_fe, dict):
                se = float(bse_fe[pred])
            else:
                # assume Series-like
                se = float(bse_fe.loc[pred])
        except Exception:
            # last resort: try to get from model_output.bse if it exists and is index-compatible
            se = None

        # compute z (or t) and p-value
        z = None
        p = None
        if se is not None and se != 0:
            z = coef / se
            # try to get p-value directly
            try:
                if pvalues is not None:
                    # pvalues might be a Series-like
                    p = float(pvalues[pred])
                else:
                    # compute two-sided p from normal distribution using math.erfc
                    p = float(math.erfc(abs(z) / math.sqrt(2)))
            except Exception:
                # fallback to erfc computation
                p = float(math.erfc(abs(z) / math.sqrt(2)))
        else:
            # If se missing, try to extract p from pvalues directly
            try:
                if pvalues is not None and pred in pvalues:
                    p = float(pvalues[pred])
            except Exception:
                p = None

        # Confidence intervals
        ci_lower = None
        ci_upper = None
        if conf_int is not None:
            try:
                # conf_int may be array-like with index
                if hasattr(conf_int, 'loc'):
                    ci_lower = float(conf_int.loc[pred, 0])
                    ci_upper = float(conf_int.loc[pred, 1])
                else:
                    # Try dict-like access
                    ci = conf_int[pred]
                    ci_lower = float(ci[0])
                    ci_upper = float(ci[1])
            except Exception:
                ci_lower = None
                ci_upper = None
        else:
            if se is not None:
                ci_lower = float(coef - 1.96 * se)
                ci_upper = float(coef + 1.96 * se)

        # exponentiated coefficient: multiplicative effect on efficiency (since DV is log(nuts/second))
        try:
            exp_coef = float(math.exp(coef))
        except Exception:
            exp_coef = None

        results[pred] = {
            "available": True,
            "coef": coef,
            "se": se,
            "z_or_t": z,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "exp_coef": exp_coef,
            "interpretation_note": (
                "The model predicts log-efficiency (log(nuts_opened / seconds)). "
                "A coefficient b means a multiplicative factor exp(b) on raw efficiency per unit change in predictor."
            )
        }

    # Build a concise human-readable description
    lines = []
    lines.append("Summary of effects on nut-cracking efficiency (DV = log(nuts_opened / seconds)), controlling for hammer and random intercept for chimpanzee:")
    for pred, stats in results.items():
        if not stats.get("available", False):
            lines.append(f"- {pred}: NOT ESTIMATED ({stats.get('message')})")
            continue
        coef = stats["coef"]
        p = stats["p_value"]
        expc = stats["exp_coef"]
        ci_l = stats["ci_lower"]
        ci_u = stats["ci_upper"]

        # significance statement
        sig = ""
        if p is None:
            sig = "p-value not available."
        else:
            if p < 0.001:
                sig = f"highly significant (p < 0.001)."
            elif p < 0.01:
                sig = f"statistically significant (p = {p:.3g})."
            elif p < 0.05:
                sig = f"statistically significant (p = {p:.3g})."
            else:
                sig = f"not statistically significant (p = {p:.3g})."

        # direction
        if coef > 0:
            direction = "associated with higher log-efficiency (positive effect)."
        elif coef < 0:
            direction = "associated with lower log-efficiency (negative effect)."
        else:
            direction = "no directional effect (coef = 0)."

        # multiplicative interpretation
        mult = ""
        if expc is not None:
            mult = f"Exp(coef) = {expc:.3f}, i.e. multiplicative factor on raw efficiency."

        ci_str = ""
        if ci_l is not None and ci_u is not None:
            ci_str = f"95% CI for coef: [{ci_l:.3f}, {ci_u:.3f}]."

        lines.append(f"- {pred}: coef = {coef:.4f}; {sig} {direction} {ci_str} {mult}")

    description = " ".join(lines)

    return {"object": results, "description": description}