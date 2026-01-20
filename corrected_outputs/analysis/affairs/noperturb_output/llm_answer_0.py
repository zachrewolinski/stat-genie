def extract_final_answer(model_output):
    """
    Extracts statistics for the 'Children' coefficient from the provided model_output.
    Expects model_output to be a dict containing keys 'nb_model' and 'ols_model'
    with fitted statsmodels result objects (GLMResultsWrapper and RegressionResultsWrapper).
    Returns a dict with keys:
      - "object": a dict containing extracted numeric results for both models and a short conclusion
      - "description": a short textual interpretation of what the numbers mean
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict with keys 'nb_model' and 'ols_model'")

    if 'nb_model' not in model_output or 'ols_model' not in model_output:
        raise KeyError("model_output must contain 'nb_model' and 'ols_model' keys")

    nb = model_output['nb_model']
    ols = model_output['ols_model']

    var = 'Children'

    def safe_get_params(mod, varname):
        # Get coef, se, pvalue
        try:
            coef = float(mod.params[varname])
        except Exception:
            raise KeyError(f"Coefficient for '{varname}' not found in model params")
        try:
            se = float(mod.bse[varname])
        except Exception:
            se = None
        try:
            pval = float(mod.pvalues[varname])
        except Exception:
            pval = None
        # Confidence interval extraction with fallbacks
        try:
            ci_table = mod.conf_int()
            # conf_int may be a DataFrame or ndarray
            if hasattr(ci_table, 'loc'):
                ci_low, ci_high = ci_table.loc[varname].values
            else:
                # ndarray: rows align with params order
                names = list(mod.params.index)
                idx = names.index(varname)
                ci_low, ci_high = ci_table[idx]
        except Exception:
            ci_low = ci_high = None
        return {"coef": coef, "se": se, "pval": pval, "ci_low": ci_low, "ci_high": ci_high}

    nb_stats = safe_get_params(nb, var)
    ols_stats = safe_get_params(ols, var)

    # For count model (NB with log link), transform coef to incidence rate ratio (IRR)
    try:
        irr = float(np.exp(nb_stats["coef"]))
    except Exception:
        irr = None
    try:
        irr_ci_low = float(np.exp(nb_stats["ci_low"])) if nb_stats["ci_low"] is not None else None
        irr_ci_high = float(np.exp(nb_stats["ci_high"])) if nb_stats["ci_high"] is not None else None
    except Exception:
        irr_ci_low = irr_ci_high = None

    # Percent change interpretation (approx)
    try:
        pct_change = (irr - 1.0) * 100.0 if irr is not None else None
    except Exception:
        pct_change = None

    # Decision rule (using NB as primary): significance at alpha=0.05
    significance = None
    if nb_stats["pval"] is not None:
        significance = nb_stats["pval"] < 0.05

    if significance is True:
        if nb_stats["coef"] < 0:
            conclusion = (
                "Having children is associated with a statistically significant decrease "
                "in the reported frequency of extramarital affairs (Negative Binomial model)."
            )
        else:
            conclusion = (
                "Having children is associated with a statistically significant increase "
                "in the reported frequency of extramarital affairs (Negative Binomial model)."
            )
    elif significance is False:
        conclusion = (
            "There is no statistically significant association between having children "
            "and reported frequency of extramarital affairs in the Negative Binomial model (p >= 0.05)."
        )
    else:
        conclusion = "Could not determine statistical significance for the 'Children' coefficient."

    # Assemble object to return
    result_object = {
        "nb_model": {
            "coef_log_count": nb_stats["coef"],           # log-scale coefficient
            "se": nb_stats["se"],
            "p_value": nb_stats["pval"],
            "ci_95_log": [nb_stats["ci_low"], nb_stats["ci_high"]],
            "incidence_rate_ratio (IRR)": irr,
            "IRR_95_CI": [irr_ci_low, irr_ci_high],
            "percent_change_in_incidence": pct_change
        },
        "ols_model": {
            "coef": ols_stats["coef"],
            "se": ols_stats["se"],
            "p_value": ols_stats["pval"],
            "ci_95": [ols_stats["ci_low"], ols_stats["ci_high"]]
        },
        "conclusion": conclusion,
        "nb_primary_significant_at_0.05": bool(significance) if significance is not None else None
    }

    # Short human-readable description
    if result_object["nb_model"]["p_value"] is not None:
        ptxt = f"NB p-value = {result_object['nb_model']['p_value']:.4f}"
    else:
        ptxt = "NB p-value = NA"
    if result_object["nb_model"]["incidence_rate_ratio (IRR)"] is not None:
        irr_txt = f"IRR = {result_object['nb_model']['incidence_rate_ratio (IRR)']:.3f}"
    else:
        irr_txt = "IRR = NA"

    description = (
        f"Primary model: Negative Binomial regression. {ptxt}, {irr_txt}. "
        f"{conclusion} "
        "IRR below 1 indicates a reduction in expected count of affairs for respondents with children; "
        "IRR above 1 indicates an increase. OLS results are reported for robustness but NB is preferred for count data."
    )

    return {"object": result_object, "description": description}