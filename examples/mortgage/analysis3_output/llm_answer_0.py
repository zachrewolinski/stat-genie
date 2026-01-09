def extract_final_answer(model_output):
    """
    Extract effect of 'Female' from a fitted statsmodels Logit-like result.

    Returns a dict with keys:
      - "object": dict with extracted numbers: coef, odds_ratio, p_value (if available/approximated),
                  conf_int (2-tuple) if available/approximated, method used.
      - "description": short human-readable interpretation of the Female effect
                       in context (direction, magnitude, and statistical significance).
    The function tries multiple fallbacks:
      1) use model_output.params, pvalues, conf_int if available;
      2) attempt to refit an unregularized Logit on model_output.model.exog/endog to obtain p-values;
      3) approximate standard errors via Hessian (plug-in) using the returned params.
    """
    import numpy as np
    import math

    def normal_cdf(x):
        # stable normal cdf using math.erf (no scipy dependency)
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Helper to get param by name with a few fallbacks
    def get_param(params, names, target='Female'):
        # params may be pandas Series or ndarray
        try:
            # if params has getitem by name (Series)
            return float(params[target])
        except Exception:
            # try case-insensitive match among names
            if names is None:
                # if no names, and params is 1D array, cannot locate by name
                raise KeyError(f"Parameter '{target}' not found and no names available.")
            for i, n in enumerate(names):
                if n.lower() == target.lower():
                    return float(np.asarray(params)[i])
            # try substring match
            for i, n in enumerate(names):
                if target.lower() in n.lower():
                    return float(np.asarray(params)[i])
            raise KeyError(f"Parameter '{target}' not found in parameter names: {names}")

    # Prepare outputs
    result_obj = {}
    used_method = None

    # Try to obtain params and names
    params = None
    names = None
    try:
        params = model_output.params
    except Exception:
        # some wrappers expose .params as attribute under different name
        try:
            params = getattr(model_output, 'b', None)  # fallback
        except Exception:
            params = None

    try:
        names = getattr(model_output.model, 'exog_names', None)
    except Exception:
        names = None

    # Extract coefficient for Female
    try:
        coef_f = get_param(params, names, target='Female')
    except Exception as e:
        # if cannot find, return informative error-like object
        return {
            "object": None,
            "description": f"Could not locate coefficient for 'Female' in model_output. Error: {e}"
        }

    # Odds ratio
    odds_ratio = float(math.exp(coef_f))

    # Try to get p-value and confidence interval directly
    p_value = None
    conf_int = None

    # 1) If the result object has pvalues and conf_int, use them
    try:
        if hasattr(model_output, 'pvalues') and model_output.pvalues is not None:
            try:
                p_value = float(model_output.pvalues['Female'])
                used_method = "pvalues_from_result"
            except Exception:
                # try name matching
                try:
                    # convert to dict-like
                    pv = dict(zip(model_output.model.exog_names, model_output.pvalues))
                    p_value = float(pv[next(n for n in pv.keys() if n.lower() == 'female')])
                    used_method = "pvalues_from_result"
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if hasattr(model_output, 'conf_int') and callable(model_output.conf_int):
            try:
                ci = model_output.conf_int()
                # conf_int may be DataFrame or ndarray; get rows by name/index
                if hasattr(ci, 'loc'):
                    ci_row = ci.loc['Female'].tolist()
                else:
                    # assume same order as exog_names
                    names = model_output.model.exog_names
                    idx = names.index('Female') if 'Female' in names else next(i for i, n in enumerate(names) if n.lower() == 'female')
                    ci_row = ci[idx].tolist() if ci.ndim == 2 and ci.shape[0] > idx else list(ci)
                conf_int = (float(ci_row[0]), float(ci_row[1]))
                if used_method is None:
                    used_method = "conf_int_from_result"
            except Exception:
                pass
    except Exception:
        pass

    # 2) If p_value or conf_int not available, try to refit an unregularized Logit using stored model.exog/endog
    if (p_value is None or conf_int is None):
        try:
            import statsmodels.api as sm
            # get exog and endog from the stored model if possible
            exog = getattr(model_output.model, 'exog', None)
            endog = getattr(model_output.model, 'endog', None)
            exog_names = getattr(model_output.model, 'exog_names', None)
            if exog is not None and endog is not None:
                # try refit unregularized to obtain standard errors/pvalues
                try:
                    unreg = sm.Logit(endog, exog)
                    unres = unreg.fit(disp=False, method='newton')
                    # extract pvalue and conf_int if available
                    try:
                        p_value = float(unres.pvalues['Female'])
                    except Exception:
                        # find by name
                        if exog_names is not None:
                            for i, n in enumerate(exog_names):
                                if n.lower() == 'female':
                                    p_value = float(unres.pvalues[i])
                                    break
                    try:
                        ci = unres.conf_int()
                        if hasattr(ci, 'loc'):
                            conf_int = tuple(ci.loc['Female'].astype(float).tolist())
                        else:
                            if exog_names is not None:
                                idx = exog_names.index('Female') if 'Female' in exog_names else next(i for i, n in enumerate(exog_names) if n.lower() == 'female')
                                conf_int = (float(ci[idx, 0]), float(ci[idx, 1]))
                            else:
                                # fallback: use params +/- 1.96*bse
                                bse = unres.bse
                                if hasattr(bse, 'iloc'):
                                    b = float(bse['Female'])
                                else:
                                    # assume same ordering
                                    idx = 0
                                    conf_int = None
                                if conf_int is None:
                                    coef = get_param(unres.params, exog_names, 'Female')
                                    conf_int = (coef - 1.96 * float(b), coef + 1.96 * float(b))
                    except Exception:
                        pass
                    if used_method is None:
                        used_method = "refit_unregularized"
                except Exception:
                    # unregularized refit failed (e.g., perfect separation) -> will use Hessian approximation below
                    pass
        except Exception:
            # statsmodels import failed or other error; proceed
            pass

    # 3) If still missing p_value or conf_int, approximate using observed information (Hessian) with the regularized params
    if p_value is None or conf_int is None:
        try:
            # obtain exog and compute mu = logistic(X beta)
            exog = getattr(model_output.model, 'exog', None)
            endog = getattr(model_output.model, 'endog', None)
            if exog is not None:
                # ensure params vector aligns with exog columns
                beta = np.asarray(model_output.params, dtype=float)
                X = np.asarray(exog, dtype=float)
                # linear predictor
                linpred = X.dot(beta)
                # logistic mean
                mu = 1.0 / (1.0 + np.exp(-linpred))
                # weight matrix diagonal
                W = mu * (1.0 - mu)
                # compute Hessian approximation H = - X^T W X
                # form X^T * (W[:,None] * X)
                WX = X * W[:, None]
                H = - X.T.dot(WX)
                # invert Hessian (use pseudo-inverse if necessary)
                try:
                    cov = np.linalg.inv(-H)
                except Exception:
                    cov = np.linalg.pinv(-H)
                # locate index of Female
                exog_names = getattr(model_output.model, 'exog_names', None)
                if exog_names is None:
                    # cannot identify index
                    raise RuntimeError("Cannot identify index of 'Female' for Hessian-based SE (no exog_names).")
                idx = None
                for i, n in enumerate(exog_names):
                    if n.lower() == 'female':
                        idx = i
                        break
                if idx is None:
                    # substring match
                    for i, n in enumerate(exog_names):
                        if 'female' in n.lower():
                            idx = i
                            break
                if idx is None:
                    raise RuntimeError("Could not find 'Female' in exog_names for Hessian-based SE.")
                se = float(math.sqrt(max(0.0, cov[idx, idx])))
                # wald z and p
                z = coef_f / se if se > 0 else float('nan')
                p_value_approx = 2.0 * (1.0 - normal_cdf(abs(z)))
                p_value = float(p_value_approx)
                # confidence interval approximate
                conf_int = (float(coef_f - 1.96 * se), float(coef_f + 1.96 * se))
                if used_method is None:
                    used_method = "hessian_approximation"
        except Exception:
            # give up on approximation if anything fails
            pass

    # Final assembly: create object and human-readable description
    extracted = {
        "coef": float(coef_f),
        "odds_ratio": float(odds_ratio),
        "method_used": used_method or "params_only"
    }
    if p_value is not None:
        extracted["p_value"] = float(p_value)
    if conf_int is not None:
        extracted["conf_int"] = (float(conf_int[0]), float(conf_int[1]))

    # Interpretation text
    # Determine significance level phrasing if p_value available
    sig_text = "statistical significance could not be determined"
    if "p_value" in extracted and extracted["p_value"] is not None and not (math.isnan(extracted["p_value"])):
        pv = extracted["p_value"]
        if pv < 0.001:
            sig_text = "highly statistically significant (p < 0.001)"
        elif pv < 0.01:
            sig_text = "statistically significant (p < 0.01)"
        elif pv < 0.05:
            sig_text = "statistically significant (p < 0.05)"
        else:
            sig_text = f"not statistically significant (p = {pv:.3g})"

    # Direction interpretation
    if coef_f > 0:
        direction = "Female applicants are associated with higher log-odds of mortgage approval (positive coefficient)."
    elif coef_f < 0:
        direction = "Female applicants are associated with lower log-odds of mortgage approval (negative coefficient)."
    else:
        direction = "No estimated difference in log-odds of approval by gender (coefficient is 0)."

    desc_lines = [
        f"Estimated coefficient for Female = {coef_f:.4g}; odds ratio = {odds_ratio:.4g}.",
        direction,
        f"Significance: {sig_text}.",
        "Model includes controls for BadHistory, Married, SelfEmployed, PI_ratio_z, Denied_PMI_z, LoanToValue, HousingExpenseRatio_z, and CreditScore_z.",
        f"Estimation method used for inference: {used_method or 'direct params (no p-value/confint available)'}."
    ]
    description = " ".join(desc_lines)

    return {"object": extracted, "description": description}