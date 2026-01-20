def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderView (and its interaction with DyslexiaBin)
    from a fitted statsmodels RegressionResultsWrapper and returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, p-values, 95% CIs,
                  and multiplicative effects (exp(coef)) for:
                    * main ReaderView effect (baseline / reference dyslexia group)
                    * ReaderView effect for DyslexiaBin = 0, 1, 2 (marginal effects)
      - "description": a short plain-language interpretation of the extracted numbers.

    The function attempts to use the model's t_test to get inference for linear
    combinations (needed to obtain ReaderView effect within each dyslexia subgroup).
    It is robust to missing interaction parameters (assumes 0 if an interaction term
    for a subgroup is not present).
    """
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing .params).")

    param_names = list(res.params.index)

    # Helper: run a linear test for a specified linear combination of parameters
    def _lincomb_test(weights):
        # weights: dict param_name -> weight (float)
        L = np.zeros(len(param_names))
        for name, w in weights.items():
            if name not in param_names:
                # treat missing param as weight 0 (no contribution)
                continue
            L[param_names.index(name)] = w
        t_res = res.t_test(L)
        # effect, se, pvalue, conf_int
        est = float(np.atleast_1d(t_res.effect)[0])
        se = float(np.atleast_1d(t_res.sd)[0])
        pval = float(np.atleast_1d(t_res.pvalue)[0])
        ci = np.atleast_2d(t_res.conf_int())[0].tolist()
        return {"coef": est, "se": se, "pvalue": pval, "ci_95": ci}

    # Identify main ReaderView parameter name (likely exactly 'ReaderView')
    main_name = None
    for n in param_names:
        if n == "ReaderView":
            main_name = n
            break
    if main_name is None:
        # Fallback: any parameter that equals 'ReaderView' ignoring possible formatting
        for n in param_names:
            if n.strip().startswith("ReaderView") and ":" not in n:
                main_name = n
                break
    if main_name is None:
        raise ValueError("Could not find a parameter named 'ReaderView' in the model parameters.")

    # Identify interaction terms for dyslexia groups (look for patterns containing both ReaderView and DyslexiaBin)
    inter_terms = [n for n in param_names if ("ReaderView" in n) and ("DyslexiaBin" in n)]
    # Map expected labels T.1 and T.2 if present
    inter_T1 = next((n for n in inter_terms if "T.1" in n or "[T.1]" in n), None)
    inter_T2 = next((n for n in inter_terms if "T.2" in n or "[T.2]" in n), None)

    # Main (baseline) effect: ReaderView (this corresponds to DyslexiaBin reference group, typically 0)
    main_test = _lincomb_test({main_name: 1})
    main_test["exp_coef"] = np.exp(main_test["coef"])

    # Effects by dyslexia group:
    # DyslexiaBin == 0 (reference): same as main
    dys0 = dict(main_test)

    # DyslexiaBin == 1: ReaderView + ReaderView:C(DyslexiaBin)[T.1] (if interaction exists)
    weights_1 = {main_name: 1}
    if inter_T1:
        weights_1[inter_T1] = 1
    dys1_test = _lincomb_test(weights_1)
    dys1_test["exp_coef"] = np.exp(dys1_test["coef"])

    # DyslexiaBin == 2: ReaderView + ReaderView:C(DyslexiaBin)[T.2] (if interaction exists)
    weights_2 = {main_name: 1}
    if inter_T2:
        weights_2[inter_T2] = 1
    dys2_test = _lincomb_test(weights_2)
    dys2_test["exp_coef"] = np.exp(dys2_test["coef"])

    # Also collect raw parameter table for quick reference for the ReaderView and interaction rows (if present)
    param_table = {}
    for n in [main_name, inter_T1, inter_T2]:
        if n and n in param_names:
            param_table[n] = {
                "coef": float(res.params[n]),
                "se": float(res.bse[n]) if hasattr(res, "bse") else None,
                "pvalue": float(res.pvalues[n]) if hasattr(res, "pvalues") else None,
                "ci_95": None
            }
            # attempt CI from conf_int
            try:
                ci = res.conf_int().loc[n].tolist()
                param_table[n]["ci_95"] = ci
            except Exception:
                param_table[n]["ci_95"] = None

    output = {
        "object": {
            "readerview_param_name": main_name,
            "interaction_param_names": {"T.1": inter_T1, "T.2": inter_T2},
            "raw_params_of_interest": param_table,
            "effects": {
                "dyslexia_0 (reference)": dys0,
                "dyslexia_1": dys1_test,
                "dyslexia_2": dys2_test
            }
        },
        "description": (
            "Extracted estimated effects of ReaderView (on log reading speed). Each 'coef' is the "
            "estimated change in log(words/sec) associated with turning ReaderView ON for the given "
            "dyslexia subgroup; 'exp_coef' is the multiplicative change in reading speed (words/sec). "
            "P-values and 95% CIs are reported for each subgroup effect. "
            "To conclude whether ReaderView 'improves' reading speed for people with dyslexia, inspect "
            "the coef and pvalue for 'dyslexia_1' (DyslexiaBin==1) and 'dyslexia_2' (DyslexiaBin==2): "
            "a positive coef and p < 0.05 indicates a statistically significant increase in reading speed "
            "for that subgroup. The returned 'object' contains these numbers programmatically."
        )
    }

    return output