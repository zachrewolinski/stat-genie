def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, odds ratios and 95% CIs for the main predictors
    from a statsmodels-like logistic regression results object or from the placeholder
    namespace returned when the dependent variable has a single class.

    Returns a dict with keys:
      - "object": dict keyed by predictor names with extracted numeric results (or None)
      - "description": human-readable explanation of what the extracted values mean

    Handles cases where params are all NaN (placeholder), or where a real fitted
    results object is provided.
    """
    import numpy as np
    import pandas as pd
    import math

    # Prepare safe access helpers
    def _as_series(x):
        if x is None:
            return None
        if isinstance(x, pd.Series):
            return x
        try:
            return pd.Series(x)
        except Exception:
            return pd.Series(list(x))

    try:
        params = getattr(model_output, "params", None)
        bse = getattr(model_output, "bse", None)
        pvalues = getattr(model_output, "pvalues", None)
        nobs = getattr(model_output, "nobs", None)
        converged = getattr(model_output, "converged", None)

        params_s = _as_series(params)
        bse_s = _as_series(bse)
        pvalues_s = _as_series(pvalues)

        # If params are missing entirely
        if params_s is None:
            return {
                "object": None,
                "description": "Model output does not contain 'params'. Cannot extract estimates."
            }

        # If all params are NaN -> placeholder case (e.g., single-class outcome or failed fit)
        if params_s.isna().all():
            # Try to infer the constant prediction (class) from the predict function if present
            const_pred = None
            pred_fn = getattr(model_output, "predict", None)
            if callable(pred_fn):
                try:
                    p = pred_fn(1)  # placeholder predict expects a row-count or exog; it will return an array/constant
                    if hasattr(p, "__iter__"):
                        const_pred = float(p[0])
                    else:
                        const_pred = float(p)
                except Exception:
                    const_pred = None

            desc = (
                "Model did not produce estimable coefficients (all parameter estimates are NaN). "
                f"This commonly occurs when the dependent variable contains a single class or the fit failed. "
                f"nobs={nobs}, converged={converged}."
            )
            if const_pred is not None:
                desc += f" The model's predict function returns constant class {const_pred}, indicating all observed contests had that outcome. No inference about predictors is possible."

            return {
                "object": {
                    "params": None,
                    "pvalues": None,
                    "odds_ratios": None,
                    "nobs": int(nobs) if (nobs is not None and not pd.isna(nobs)) else None,
                    "converged": bool(converged) if converged is not None else None,
                    "constant_predicted_class": const_pred
                },
                "description": desc
            }

        # Normal case: compute odds ratios and 95% CIs where possible
        # Align index names if available
        try:
            params_s.index = params_s.index.astype(str)
        except Exception:
            pass
        if bse_s is not None:
            try:
                bse_s.index = bse_s.index.astype(str)
            except Exception:
                pass
        if pvalues_s is not None:
            try:
                pvalues_s.index = pvalues_s.index.astype(str)
            except Exception:
                pass

        coefs = params_s.to_dict()
        bses = bse_s.to_dict() if bse_s is not None else {}
        pvals = pvalues_s.to_dict() if pvalues_s is not None else {}

        def _safe_exp(x):
            try:
                return float(np.exp(x))
            except Exception:
                return None

        def _ci_from_coef_se(coef, se, z=1.96):
            if coef is None or se is None or (isinstance(coef, float) and np.isnan(coef)) or (isinstance(se, float) and np.isnan(se)):
                return (None, None)
            lo = np.exp(coef - z * se)
            hi = np.exp(coef + z * se)
            return (float(lo), float(hi))

        # Focus on the predictors named in the task
        predictors = ['RelSizeRatio_z', 'ContestLocation_FocalHome', 'ContestLocation_OtherHome']
        extracted = {}
        for pred in predictors:
            coef = coefs.get(pred, None)
            pv = pvals.get(pred, None)
            se = bses.get(pred, None)
            or_ = _safe_exp(coef) if (coef is not None and not (isinstance(coef, float) and np.isnan(coef))) else None
            ci = _ci_from_coef_se(coef, se)
            extracted[pred] = {
                "coef": None if (coef is None or (isinstance(coef, float) and np.isnan(coef))) else float(coef),
                "pvalue": None if (pv is None or (isinstance(pv, float) and np.isnan(pv))) else float(pv),
                "odds_ratio": or_,
                "CI_95": ci
            }

        # Build a compact human-readable description
        lines = []
        for pred in predictors:
            info = extracted[pred]
            if info["coef"] is None:
                lines.append(f"{pred}: estimate not available.")
            else:
                pv_txt = "p unavailable" if info["pvalue"] is None else f"p = {info['pvalue']:.3g}"
                ci_txt = f"95% CI for OR = ({info['CI_95'][0]:.3g}, {info['CI_95'][1]:.3g})" if (info['CI_95'][0] is not None) else "95% CI unavailable"
                lines.append(
                    f"{pred}: coef = {info['coef']:.3f}, OR = {info['odds_ratio']:.3f}, {ci_txt}, {pv_txt}"
                )

        description = (
            f"Extracted estimates for the main predictors (n = {int(nobs) if nobs is not None else 'unknown'}, converged = {converged}). "
            + " ".join(lines)
            + " Interpretation: a positive coefficient (OR>1) means higher predictor value is associated with higher probability that the focal group wins; "
            + "a negative coefficient (OR<1) means the opposite. Statistical significance should be judged from the p-values and CIs above."
        )

        return {
            "object": {
                "predictor_estimates": extracted,
                "nobs": int(nobs) if (nobs is not None and not pd.isna(nobs)) else None,
                "converged": bool(converged) if converged is not None else None,
                "full_params": coefs,
                "full_pvalues": pvals,
                "full_bse": bses
            },
            "description": description
        }

    except Exception as exc:
        return {
            "object": None,
            "description": f"An error occurred while extracting information from model_output: {exc}"
        }