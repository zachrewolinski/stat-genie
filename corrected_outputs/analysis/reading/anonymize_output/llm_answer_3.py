def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of ReaderView on ReadingSpeed_wps,
    including the interaction with Dyslexia, from a fitted statsmodels OLSResults
    (robust) object.

    Returns a dictionary:
      {
        "object": { detailed numeric results... },
        "description": "Plain-language interpretation"
      }
    """
    import numpy as np

    # Obtain params array-like and attempt to get parameter names robustly
    params = getattr(model_output, "params", None)

    # Try common locations for parameter names in statsmodels results
    param_names = None
    # 1) model.exog_names
    if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
        try:
            param_names = list(model_output.model.exog_names)
        except Exception:
            param_names = None
    # 2) param_names attribute on result
    if param_names is None and hasattr(model_output, "param_names"):
        try:
            param_names = list(model_output.param_names)
        except Exception:
            param_names = None
    # 3) params index (pandas Series)
    if param_names is None and params is not None and hasattr(params, "index"):
        try:
            param_names = list(params.index)
        except Exception:
            param_names = None
    # 4) fallback: if params is array-like, create generic names
    if param_names is None:
        # determine length k if possible
        k = None
        if params is not None:
            try:
                k = int(np.asarray(params).shape[0])
            except Exception:
                k = None
        if k is None and hasattr(model_output, "k_params"):
            try:
                k = int(model_output.k_params)
            except Exception:
                k = None
        if k is None:
            raise ValueError("Could not determine parameter names or number of parameters from model_output.")
        param_names = [f"beta{i}" for i in range(k)]

    # Ensure params is a 1-d numpy array for numeric access
    params_arr = None
    if params is None:
        # try to construct from model_output.params if possible
        raise ValueError("model_output.params is missing.")
    else:
        params_arr = np.asarray(params).ravel()

    # Helper to convert an array-like of values into a name->value dict
    def values_to_dict(values, names):
        a = np.asarray(values).ravel()
        if a.shape[0] != len(names):
            # if shapes mismatch, try to pad/trim as necessary
            if a.shape[0] < len(names):
                # pad with NaN
                padded = np.full((len(names),), np.nan, dtype=float)
                padded[: a.shape[0]] = a
                a = padded
            else:
                a = a[: len(names)]
        return {n: float(a[i]) if not np.isnan(a[i]) else None for i, n in enumerate(names)}

    # Build basic results dict with params, pvalues, and bse
    results = {}
    results["param_names"] = list(param_names)
    results["params"] = values_to_dict(params_arr, param_names)

    # pvalues
    pvalues = getattr(model_output, "pvalues", None)
    if pvalues is not None:
        try:
            results["pvalues"] = values_to_dict(np.asarray(pvalues).ravel(), param_names)
        except Exception:
            results["pvalues"] = None
    else:
        results["pvalues"] = None

    # bse (standard errors)
    bse = getattr(model_output, "bse", None)
    if bse is not None:
        try:
            results["bse"] = values_to_dict(np.asarray(bse).ravel(), param_names)
        except Exception:
            results["bse"] = None
    else:
        results["bse"] = None

    # conf_int for each parameter
    conf_int_raw = None
    try:
        conf_int_raw = model_output.conf_int()
    except Exception:
        conf_int_raw = None

    conf_int_dict = {}
    if conf_int_raw is not None:
        try:
            # conf_int_raw might be a DataFrame-like with index matching param names
            # or a numpy array with shape (k, 2)
            if hasattr(conf_int_raw, "loc"):
                # DataFrame-like
                for n in param_names:
                    try:
                        row = conf_int_raw.loc[n]
                        conf_int_dict[n] = (float(row.iloc[0]), float(row.iloc[1]))
                    except Exception:
                        # fallback: if name not found, try numeric index
                        pass
                # if any names missing, try mapping by row order
                missing = [n for n in param_names if n not in conf_int_dict]
                if missing:
                    try:
                        arr = np.asarray(conf_int_raw)
                        for i, n in enumerate(param_names):
                            if n in conf_int_dict:
                                continue
                            if i < arr.shape[0]:
                                conf_int_dict[n] = (float(arr[i, 0]), float(arr[i, 1]))
                    except Exception:
                        pass
            else:
                arr = np.asarray(conf_int_raw)
                for i, n in enumerate(param_names):
                    if i < arr.shape[0]:
                        conf_int_dict[n] = (float(arr[i, 0]), float(arr[i, 1]))
        except Exception:
            conf_int_dict = {}
    results["conf_int"] = conf_int_dict if conf_int_dict else None

    # Helper to get parameter index by name
    name_to_idx = {n: i for i, n in enumerate(param_names)}

    # Helper to find parameter names robustly
    def find_param(name):
        # exact match first
        if name in param_names:
            return name
        # fall back to containing tokens (useful if patsy produced different formatting)
        tokens = name.split(":")
        for n in param_names:
            if all(tok in n for tok in tokens):
                return n
        return None

    name_reader = find_param("ReaderView")
    name_inter = None
    # interaction may be named 'ReaderView:Dyslexia' or similar
    for n in param_names:
        if "ReaderView" in n and "Dyslexia" in n and n != name_reader:
            name_inter = n
            break
    # If explicit search didn't find, try the expected exact label
    if name_inter is None:
        name_inter = find_param("ReaderView:Dyslexia")

    if name_reader is None:
        raise ValueError("Could not find a parameter corresponding to 'ReaderView' in the model parameters.")

    k = len(param_names)

    def contrast_for(effect_at_dyslexia_value):
        # Create contrast vector for effect: ReaderView + effect_at_dyslexia_value * (ReaderView:Dyslexia)
        c = np.zeros((k,), dtype=float)
        c[name_to_idx[name_reader]] = 1.0
        if name_inter is not None:
            c[name_to_idx[name_inter]] = float(effect_at_dyslexia_value)
        return c

    # Marginal effect when Dyslexia = 0
    c0 = contrast_for(0)
    try:
        t0 = model_output.t_test(c0)
        eff0 = float(np.asarray(t0.effect).ravel()[0])
        se0 = float(np.asarray(t0.sd).ravel()[0])
        tval0 = float(np.asarray(t0.tvalue).ravel()[0])
        pval0 = float(np.asarray(t0.pvalue).ravel()[0])
        try:
            ci0_arr = np.asarray(t0.conf_int(alpha=0.05))
            ci0 = (float(ci0_arr[0, 0]), float(ci0_arr[0, 1]))
        except Exception:
            ci0 = None
    except Exception:
        eff0 = se0 = tval0 = pval0 = None
        ci0 = None

    # Marginal effect when Dyslexia = 1
    c1 = contrast_for(1)
    try:
        t1 = model_output.t_test(c1)
        eff1 = float(np.asarray(t1.effect).ravel()[0])
        se1 = float(np.asarray(t1.sd).ravel()[0])
        tval1 = float(np.asarray(t1.tvalue).ravel()[0])
        pval1 = float(np.asarray(t1.pvalue).ravel()[0])
        try:
            ci1_arr = np.asarray(t1.conf_int(alpha=0.05))
            ci1 = (float(ci1_arr[0, 0]), float(ci1_arr[0, 1]))
        except Exception:
            ci1 = None
    except Exception:
        eff1 = se1 = tval1 = pval1 = None
        ci1 = None

    results["marginal_effect_dyslexia_0"] = {
        "estimate": eff0,
        "se": se0,
        "t": tval0,
        "pvalue": pval0,
        "ci95": ci0,
        "interpretation": "Effect of ReaderView on reading speed for non-dyslexic readers (Dyslexia=0). Units: words/sec."
    }
    results["marginal_effect_dyslexia_1"] = {
        "estimate": eff1,
        "se": se1,
        "t": tval1,
        "pvalue": pval1,
        "ci95": ci1,
        "interpretation": "Effect of ReaderView on reading speed for dyslexic readers (Dyslexia=1). Units: words/sec."
    }

    # Summarize model fit and sample size
    try:
        nobs = int(model_output.nobs)
    except Exception:
        nobs = None
    try:
        r2 = float(model_output.rsquared)
        adjr2 = float(model_output.rsquared_adj)
    except Exception:
        r2 = adjr2 = None

    results["nobs"] = nobs
    results["r2"] = r2
    results["adj_r2"] = adjr2

    # Construct a concise conclusion about whether ReaderView improves reading speed for dyslexic readers.
    # Use p < 0.05 as the significance threshold and require a positive estimate to conclude "improves".
    sig_threshold = 0.05
    if (pval1 is not None) and (eff1 is not None):
        if pval1 < sig_threshold and eff1 > 0:
            conclusion = "Yes — the estimated effect of ReaderView for dyslexic readers is positive (%.4g words/sec) and statistically significant (p = %.3g)." % (eff1, pval1)
        elif pval1 < sig_threshold and eff1 <= 0:
            conclusion = "No — the estimated effect of ReaderView for dyslexic readers is statistically significant (p = %.3g) but non-positive (estimate = %.4g words/sec), so it does not improve reading speed." % (pval1, eff1)
        else:
            conclusion = "No strong evidence — the estimated effect of ReaderView for dyslexic readers is %.4g words/sec with p = %.3g, which is not statistically significant at α = %.2f." % (eff1, pval1, sig_threshold)
    else:
        conclusion = "Could not compute a reliable conclusion for the effect of ReaderView on dyslexic readers due to missing statistics."

    description = (
        "Extracted coefficients, standard errors, p-values, and 95% CIs for the ReaderView main effect "
        "and its interaction with Dyslexia. The key quantities are the marginal effects of ReaderView at "
        "Dyslexia=0 (non-dyslexic) and Dyslexia=1 (dyslexic). The conclusion below uses a two-sided p < 0.05 "
        "rule and requires a positive estimate to claim ReaderView 'improves' reading speed."
    )

    return {
        "object": {
            "results": results,
            "conclusion": conclusion
        },
        "description": description
    }