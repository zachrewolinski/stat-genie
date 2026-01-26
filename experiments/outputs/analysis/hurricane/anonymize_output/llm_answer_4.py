def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, confidence intervals, p-values, and
    interpretable effect sizes for the femininity predictors from the model_output.

    Returns a dictionary with:
      - "object": structured numeric summaries for each model and variable
      - "description": short plain-language interpretation about whether results
                       support the hypothesis that more feminine names lead to fewer fatalities.
    """
    import numpy as np
    import pandas as pd
    import math

    out = {"ols": {}, "nb": {}, "conclusion": None}

    # Helper to coerce various param-like objects into a pandas Series
    def to_series(raw, names=None):
        if raw is None:
            return None
        # If already a Series, return as-is
        if isinstance(raw, pd.Series):
            return raw
        # If dict-like
        if isinstance(raw, dict):
            try:
                return pd.Series(raw)
            except Exception:
                pass
        # If numpy array or list/tuple
        if isinstance(raw, (np.ndarray, list, tuple)):
            arr = np.array(raw)
            # If names provided and lengths match, use them
            if names is not None and len(names) == arr.shape[0]:
                try:
                    return pd.Series(arr, index=names)
                except Exception:
                    pass
            # If no names, create numeric/string index
            try:
                return pd.Series(arr, index=[str(i) for i in range(arr.shape[0])])
            except Exception:
                pass
        # Try convert directly
        try:
            return pd.Series(raw)
        except Exception:
            return None

    # Helper to extract stats from a statsmodels Results-like object
    def summarize_res(res, varname, model_type):
        if res is None:
            return None
        try:
            # Try to obtain parameter names from the model if available
            param_names = None
            if hasattr(res, "model"):
                # statsmodels common attribute
                param_names = getattr(res.model, "exog_names", None)
                # some result objects keep names in model.data.param_names
                if param_names is None:
                    model_data = getattr(res.model, "data", None)
                    if model_data is not None:
                        param_names = getattr(model_data, "param_names", None)

            raw_params = getattr(res, "params", None)
            raw_pvalues = getattr(res, "pvalues", None)
            raw_bse = getattr(res, "bse", None)
            raw_ci = None
            try:
                raw_ci = res.conf_int()
            except Exception:
                raw_ci = None

            params = to_series(raw_params, names=param_names)
            pvalues = to_series(raw_pvalues, names=param_names)
            bse = to_series(raw_bse, names=param_names)

            ci = None
            if raw_ci is not None:
                try:
                    ci_df = pd.DataFrame(raw_ci)
                    # Standardize column names
                    if ci_df.shape[1] >= 2:
                        ci_df = ci_df.iloc[:, :2]
                        ci_df.columns = ["ci_lower", "ci_upper"]
                    else:
                        ci_df.columns = ["ci_lower"]
                    # Assign index based on available names
                    if param_names is not None and len(param_names) == ci_df.shape[0]:
                        ci_df.index = param_names
                    elif params is not None and hasattr(params, "index") and len(params.index) == ci_df.shape[0]:
                        ci_df.index = params.index
                    ci = ci_df
                except Exception:
                    ci = None
        except Exception:
            return None

        # Ensure params exist and varname is present
        if params is None or varname not in params.index:
            return None

        # Extract numeric values safely
        try:
            coef = float(params[varname])
        except Exception:
            try:
                coef = float(params.loc[varname])
            except Exception:
                return None

        se = None
        try:
            if bse is not None and varname in bse.index:
                se = float(bse[varname])
            elif isinstance(bse, (int, float, np.floating, np.integer)):
                se = float(bse)
        except Exception:
            se = None

        pval = None
        try:
            if pvalues is not None and varname in pvalues.index:
                pval = float(pvalues[varname])
            elif isinstance(pvalues, (int, float, np.floating, np.integer)):
                pval = float(pvalues)
        except Exception:
            pval = None

        ci_lower = None
        ci_upper = None
        try:
            if ci is not None and varname in ci.index:
                ci_lower = float(ci.loc[varname, "ci_lower"]) if "ci_lower" in ci.columns else None
                ci_upper = float(ci.loc[varname, "ci_upper"]) if "ci_upper" in ci.columns else None
        except Exception:
            ci_lower = None
            ci_upper = None

        summary = {
            "coef": coef,
            "std_err": se,
            "p_value": pval,
            "ci_95": (ci_lower, ci_upper),
        }

        # Interpretability adjustments
        if model_type == "ols":
            # Outcome = LogDeaths. Coef on log-outcome -> multiplicative effect on Deaths.
            try:
                pct_change = (math.exp(coef) - 1.0) * 100.0
            except Exception:
                pct_change = None
            try:
                pct_ci_lower = (math.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
            except Exception:
                pct_ci_lower = None
            try:
                pct_ci_upper = (math.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
            except Exception:
                pct_ci_upper = None

            summary.update({
                "interpretation": {
                    "outcome": "LogDeaths",
                    "approx_percent_change_per_unit_in_predictor": pct_change,
                    "percent_change_95ci": (pct_ci_lower, pct_ci_upper),
                    "note": "For Femininity_z (standardized), this is percent change in expected Deaths per 1 SD increase in femininity. For FemaleNameBinary, this is percent change comparing female vs male name."
                }
            })
        elif model_type == "nb":
            # Count model: exponentiate coef -> incidence rate ratio (IRR)
            try:
                irr = math.exp(coef)
            except Exception:
                irr = None
            try:
                irr_ci_lower = math.exp(ci_lower) if ci_lower is not None else None
            except Exception:
                irr_ci_lower = None
            try:
                irr_ci_upper = math.exp(ci_upper) if ci_upper is not None else None
            except Exception:
                irr_ci_upper = None
            try:
                irr_pct = (irr - 1.0) * 100.0 if irr is not None else None
            except Exception:
                irr_pct = None
            irr_pct_ci = (
                ((irr_ci_lower - 1.0) * 100.0) if irr_ci_lower is not None else None,
                ((irr_ci_upper - 1.0) * 100.0) if irr_ci_upper is not None else None,
            )
            summary.update({
                "interpretation": {
                    "outcome": "Deaths (count model)",
                    "IRR": irr,
                    "IRR_95ci": (irr_ci_lower, irr_ci_upper),
                    "percent_change_in_rate": irr_pct,
                    "percent_change_95ci": irr_pct_ci,
                    "note": "IRR < 1 means fewer deaths associated with higher predictor; IRR > 1 means more deaths."
                }
            })
        return summary

    # Variables of interest
    vars_of_interest = ["Femininity_z", "FemaleNameBinary"]

    # Summaries for OLS (LogDeaths)
    ols_res = model_output.get("ols", None)
    for v in vars_of_interest:
        s = summarize_res(ols_res, v, model_type="ols")
        out["ols"][v] = s

    # Summaries for NB (Deaths)
    nb_res = model_output.get("nb", None)
    for v in vars_of_interest:
        s = summarize_res(nb_res, v, model_type="nb")
        out["nb"][v] = s

    # Form a simple conclusion based on sign and p-value from both models:
    # We say evidence supports hypothesis if both models show negative effect
    # (coef < 0 or IRR < 1) and at least one is statistically significant at p < 0.05,
    # and none show a significant effect in the opposite direction.
    def evidence_for_hypothesis():
        sig_negative = False
        sig_positive = False
        any_result = False

        for model_key in ["ols", "nb"]:
            for var in vars_of_interest:
                summary = out[model_key].get(var)
                if summary is None:
                    continue
                any_result = True
                coef = summary.get("coef")
                p = summary.get("p_value")
                if coef is None:
                    continue
                if coef < 0:
                    if p is not None and p < 0.05:
                        sig_negative = True
                elif coef > 0:
                    if p is not None and p < 0.05:
                        sig_positive = True
                # coef == 0 or p is None -> treated as non-significant/no-direction

        if not any_result:
            return "No model results available to evaluate the hypothesis."

        if sig_negative and not sig_positive:
            return "Evidence supports the hypothesis: more feminine names are associated with fewer fatalities (negative effect; statistically significant in at least one model and no significant positive effects)."
        if sig_positive and not sig_negative:
            return "Evidence contradicts the hypothesis: more feminine names are associated with more fatalities (positive effect; statistically significant in at least one model and no significant negative effects)."
        # Mixed or no significant results
        return "Inconclusive evidence: either effects are not statistically significant or models show mixed directions."

    out["conclusion"] = evidence_for_hypothesis()

    # Short human-readable description
    desc_lines = []
    desc_lines.append("Extracted estimates for 'Femininity_z' and 'FemaleNameBinary' from the two fitted models.")
    desc_lines.append("For OLS (LogDeaths): coefficients are changes in log(Deaths); exp(coef)-1 gives approximate percent change in Deaths.")
    desc_lines.append("For Negative Binomial (Deaths): exp(coef) gives an incidence rate ratio (IRR); IRR-1 gives percent change in expected count.")
    desc_lines.append("Summaries (numeric) are in the 'object' field. Final verdict on the hypothesis is in 'conclusion'.")

    out_verbose = {
        "object": out,
        "description": " ".join(desc_lines)
    }
    return out_verbose