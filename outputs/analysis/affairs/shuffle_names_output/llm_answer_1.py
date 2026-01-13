def extract_final_answer(model_output):
    """
    Extracts the effect of HasChildren on extramarital affairs from fitted model objects.

    Input:
      model_output: dict-like with keys (expected) 'ols_log1p', 'poisson', 'neg_binomial'
                    values are statsmodels ResultsWrapper objects (or None).

    Returns:
      dict with keys:
        - "object": a dict containing per-model extracted statistics and an overall
                    simple verdict about whether having children decreases affairs.
        - "description": brief explanation of the contents of "object" and how to
                         interpret the transformed effects.

    The extracted statistics per model include:
      - coef: raw coefficient on HasChildren
      - pvalue: p-value for that coefficient
      - ci: 95% confidence interval for the raw coefficient
      - transformed_effect: for OLS(log1p) -> percent change in (1+AffairCount) ≈ (exp(coef)-1)*100;
                            for Poisson/NegBin -> incidence rate ratio IRR = exp(coef)
      - transformed_ci: CI for transformed effect (same transform applied to CI endpoints)
      - significant: boolean whether pvalue < 0.05
      - nobs: number of observations used by that model (if available)
    """
    import numpy as np
    import pandas as pd

    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict-like object with model results.")

    models_to_check = ['ols_log1p', 'poisson', 'neg_binomial']
    results_summary = {}
    n_models_nonnull = 0
    negative_and_significant_count = 0
    negative_count = 0
    any_significant = False

    for mkey in models_to_check:
        res = model_output.get(mkey, None)
        if res is None:
            results_summary[mkey] = None
            continue

        # Try to access parameters as a pandas Series; handle discrete or GLM results
        try:
            params = res.params
            pvalues = res.pvalues
        except Exception as e:
            results_summary[mkey] = {"error": f"Could not read params/pvalues: {e}"}
            continue

        # Identify the exact parameter name corresponding to HasChildren
        param_index = None
        if isinstance(params, (pd.Series, pd.DataFrame)):
            idxs = list(params.index)
        else:
            # fallback: coerce to pandas Series
            try:
                params = pd.Series(params)
                idxs = list(params.index)
            except Exception:
                idxs = []

        # direct match first
        if 'HasChildren' in idxs:
            param_index = 'HasChildren'
        else:
            # find any index containing the substring 'HasChildren'
            matches = [i for i in idxs if 'HasChildren' in str(i)]
            if matches:
                param_index = matches[0]

        if param_index is None:
            # cannot find parameter - record and continue
            results_summary[mkey] = {"error": "Parameter 'HasChildren' not found in model parameters."}
            continue

        # extract coefficient, pvalue
        try:
            coef = float(params[param_index])
            pval = float(pvalues[param_index])
        except Exception as e:
            results_summary[mkey] = {"error": f"Failed to extract numeric coef/pvalue: {e}"}
            continue

        # extract confidence interval
        try:
            conf = res.conf_int()
            if isinstance(conf, pd.DataFrame) and param_index in conf.index:
                ci_lower, ci_upper = float(conf.loc[param_index, 0]), float(conf.loc[param_index, 1])
            else:
                # conf_int might be ndarray; find by position
                idx_pos = idxs.index(param_index) if param_index in idxs else 0
                ci_arr = np.asarray(conf)
                ci_lower, ci_upper = float(ci_arr[idx_pos, 0]), float(ci_arr[idx_pos, 1])
        except Exception:
            ci_lower, ci_upper = None, None

        # number of observations
        nobs = None
        # many statsmodels results have .nobs or res.model.endog
        if hasattr(res, 'nobs'):
            try:
                nobs = int(res.nobs)
            except Exception:
                nobs = None
        if nobs is None:
            try:
                nobs = int(getattr(res.model, 'nobs', None))
            except Exception:
                nobs = None
        if nobs is None:
            try:
                endog = getattr(res.model, 'endog', None)
                if endog is not None:
                    nobs = int(getattr(endog, 'shape', (None,))[0])
            except Exception:
                nobs = None

        # Transform effect for interpretation
        if mkey == 'ols_log1p':
            # dependent var is log(1 + AffairCount)
            # approximate percent change in (1 + AffairCount) for a unit change in HasChildren:
            # pct_change = (exp(coef) - 1) * 100
            try:
                transformed = (np.exp(coef) - 1.0) * 100.0
                transformed_ci = None
                if ci_lower is not None and ci_upper is not None:
                    transformed_ci = ((np.exp(ci_lower) - 1.0) * 100.0, (np.exp(ci_upper) - 1.0) * 100.0)
                transformed_name = "Pct change in (1+AffairCount)"
            except Exception:
                transformed, transformed_ci, transformed_name = None, None, "transformed"
        else:
            # Poisson and NegativeBinomial: exponentiate to get incidence rate ratio (IRR)
            try:
                irr = float(np.exp(coef))
                irr_ci = None
                if ci_lower is not None and ci_upper is not None:
                    irr_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
                transformed = irr
                transformed_ci = irr_ci
                transformed_name = "IRR (multiplicative effect on expected count)"
            except Exception:
                transformed, transformed_ci, transformed_name = None, None, "transformed"

        significant = (pval is not None) and (pval < 0.05)

        # bookkeeping for a simple overall verdict
        n_models_nonnull += 1
        if coef < 0:
            negative_count += 1
            if significant:
                negative_and_significant_count += 1
        if significant:
            any_significant = True

        results_summary[mkey] = {
            "coef": coef,
            "pvalue": pval,
            "ci_raw": (ci_lower, ci_upper),
            "transformed_effect": transformed,
            "transformed_ci": transformed_ci,
            "transformed_name": transformed_name,
            "significant": significant,
            "nobs": nobs
        }

    # Simple, transparent concluding rule:
    # - "Yes (decrease)" if majority of non-null models have negative coef and at least one model has a statistically significant negative coef.
    # - "No strong evidence" otherwise (reports direction if consistent).
    overall_conclusion = "No strong evidence either way."
    if n_models_nonnull == 0:
        overall_conclusion = "No models available to form a conclusion."
    else:
        # majority negative?
        if negative_count >= (n_models_nonnull / 2.0):
            if negative_and_significant_count >= 1:
                overall_conclusion = "Evidence consistent with having children decreasing engagement in extramarital affairs (negative coefficients; at least one statistically significant)."
            else:
                overall_conclusion = "Coefficients are generally negative (suggesting fewer affairs among those with children), but no model shows a statistically significant negative effect at p<0.05."
        else:
            # not majority negative
            if any_significant:
                # there is a significant effect but not consistently negative
                overall_conclusion = "No consistent evidence that having children decreases affairs; model results are mixed and not majority negative. At least one model shows a statistically significant effect (check the per-model results)."
            else:
                overall_conclusion = "No evidence that having children decreases engagement in extramarital affairs: coefficients are mixed and none are statistically significant."

    output_object = {
        "per_model": results_summary,
        "n_models_examined": n_models_nonnull,
        "negative_coeff_count": negative_count,
        "negative_and_significant_count": negative_and_significant_count,
        "any_significant": any_significant,
        "overall_conclusion": overall_conclusion
    }

    description = (
        "Returned object contains per-model estimates for the 'HasChildren' coefficient:\n"
        "- coef and 95% CI on the raw scale (coefficient on log(1+AffairCount) or on the linear predictor for GLMs),\n"
        "- transformed_effect: for ols_log1p this is percent change in (1+AffairCount) = (exp(coef)-1)*100; "
        "for poisson/neg_binomial this is the IRR = exp(coef),\n"
        "- p-value and significance flag (p < 0.05), and sample size if available.\n\n"
        "An overall_conclusion string gives a simple, conservative summary: it reports 'evidence of decrease' only if most models have negative coefficients and at least one is statistically significant; otherwise it reports that evidence is weak or mixed."
    )

    return {"object": output_object, "description": description}