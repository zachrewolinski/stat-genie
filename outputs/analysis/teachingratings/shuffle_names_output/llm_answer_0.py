def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, and 95% CIs for beauty measures
    from a statsmodels RegressionResultsWrapper (fitted with robust cov_type).

    Returns:
      {
        "object": {
          "nobs": int,
          "df_resid": float,
          "params": { varname: float, ... },
          "results_by_variable": {
             "BeautyIndex": {coef, se, pval, ci_lower, ci_upper, significant_0.05},
             "BeautyBinary": {...},                # if present
             "BeautyIndex_x_Male": {...}           # if present (interaction)
          },
          "marginal_effects": {
             "BeautyIndex_Female(Male=0)": {est, se, pval, ci_lower, ci_upper, significant_0.05},
             "BeautyIndex_Male(Male=1)": {est, se, pval, ci_lower, ci_upper, significant_0.05}   # if interaction exists
          }
        },
        "description": "Concise interpretation of the extracted statistics in plain language."
      }

    Notes:
      - If a variable is not present in the model, it will be omitted from the output.
      - Uses model_output.params, .pvalues, .conf_int(), and .cov_params() to compute quantities.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy import stats
    except Exception:
        # If scipy is not available, raise informative error
        raise ImportError("scipy is required for p-values/confidence intervals. Please install scipy.")

    res = model_output

    # Basic info
    try:
        params = res.params.copy()
    except Exception:
        raise ValueError("model_output does not appear to be a fitted statsmodels result with .params")

    cov = res.cov_params()
    # Ensure cov is a numpy array with index aligned to params
    # If cov is a DataFrame, convert to numpy but keep index/columns ordering consistent
    if hasattr(cov, "loc"):
        # Align to params.index
        cov = cov.reindex(index=params.index, columns=params.index).values
    else:
        cov = np.asarray(cov)

    pvalues = getattr(res, "pvalues", None)
    if pvalues is None:
        pvalues = {k: None for k in params.index}
    conf_int = None
    try:
        conf_int = res.conf_int(alpha=0.05)
    except Exception:
        conf_int = None

    nobs = int(getattr(res, "nobs", getattr(res, "model", None).nobs if getattr(res, "model", None) is not None else None))
    df_resid = float(getattr(res, "df_resid", None))

    def safe_get_param(name):
        return float(params[name]) if name in params.index else None

    def compute_ci_and_p(est, se, df=df_resid, alpha=0.05):
        if se is None or se != se:  # check NaN
            return {"se": None, "pval": None, "ci_lower": None, "ci_upper": None}
        # t-stat
        tstat = est / se if se != 0 else np.nan
        if df is None or df <= 0:
            pval = float(2 * (1.0 - stats.norm.cdf(abs(tstat)))) if not np.isnan(tstat) else None
            crit = stats.norm.ppf(1 - alpha/2)
        else:
            pval = float(2 * (1.0 - stats.t.cdf(abs(tstat), df))) if not np.isnan(tstat) else None
            crit = float(stats.t.ppf(1 - alpha/2, df))
        ci_lower = est - crit * se
        ci_upper = est + crit * se
        return {"se": float(se), "pval": float(pval) if pval is not None else None,
                "ci_lower": float(ci_lower), "ci_upper": float(ci_upper)}

    def lincomb_effect(weights):
        """
        weights: dict varname -> multiplier
        returns dict with estimate, se, pval, ci_lower, ci_upper
        """
        # build vector a aligned with params.index
        a = np.zeros(len(params.index), dtype=float)
        for i, name in enumerate(params.index):
            if name in weights:
                a[i] = float(weights[name])
        est = float(np.dot(a, params.values))
        var = float(np.dot(a, np.dot(cov, a)))
        se = sqrt(var) if var >= 0 else np.nan
        stats_dict = compute_ci_and_p(est, se)
        return {"est": float(est), "se": stats_dict["se"], "pval": stats_dict["pval"],
                "ci_lower": stats_dict["ci_lower"], "ci_upper": stats_dict["ci_upper"],
                "significant_0.05": (stats_dict["pval"] is not None and stats_dict["pval"] < 0.05)}

    # Collect variable-level reported estimates (direct coefficients)
    results_by_variable = {}
    for var in ["BeautyIndex", "BeautyBinary", "BeautyIndex_x_Male"]:
        if var in params.index:
            coef = float(params[var])
            # se from sqrt of diagonal of cov
            try:
                se = float(np.sqrt(cov[list(params.index).index(var), list(params.index).index(var)]))
            except Exception:
                se = None
            stats_dict = compute_ci_and_p(coef, se)
            results_by_variable[var] = {
                "coef": float(coef),
                "se": stats_dict["se"],
                "pval": stats_dict["pval"],
                "ci_lower": stats_dict["ci_lower"],
                "ci_upper": stats_dict["ci_upper"],
                "significant_0.05": (stats_dict["pval"] is not None and stats_dict["pval"] < 0.05)
            }

    # Marginal effects: BeautyIndex for Female (Male=0) and Male (Male=1)
    marginal_effects = {}
    # Female: effect is simply coef(BeautyIndex)
    if "BeautyIndex" in params.index:
        marginal_effects["BeautyIndex_Female(Male=0)"] = lincomb_effect({"BeautyIndex": 1.0})
    # Male: effect is coef(BeautyIndex) + coef(BeautyIndex_x_Male) if interaction exists
    if ("BeautyIndex" in params.index) and ("BeautyIndex_x_Male" in params.index):
        marginal_effects["BeautyIndex_Male(Male=1)"] = lincomb_effect({"BeautyIndex": 1.0, "BeautyIndex_x_Male": 1.0})
    elif ("BeautyIndex" in params.index) and ("Male" in params.index) and ("BeautyIndex_x_Male" not in params.index):
        # No explicit interaction term found; male effect equals female effect (no moderation)
        marginal_effects["BeautyIndex_Male(Male=1)"] = lincomb_effect({"BeautyIndex": 1.0})

    # Prepare output object
    out = {
        "nobs": nobs,
        "df_resid": df_resid,
        "params": {name: float(params[name]) for name in params.index},
        "results_by_variable": results_by_variable,
        "marginal_effects": marginal_effects
    }

    # Build a concise description interpreting the main results
    # We'll describe availability of variables and summarize primary estimates.
    desc_lines = []
    if "BeautyIndex" in results_by_variable:
        bi = results_by_variable["BeautyIndex"]
        desc_lines.append(f"BeautyIndex (continuous): coef={bi['coef']:.4f}, SE={bi['se']:.4f}, p={bi['pval']:.4g}, 95%CI=[{bi['ci_lower']:.4f}, {bi['ci_upper']:.4f}].")
    else:
        desc_lines.append("BeautyIndex not included in the model output.")

    if "BeautyBinary" in results_by_variable:
        bb = results_by_variable["BeautyBinary"]
        desc_lines.append(f"BeautyBinary (indicator): coef={bb['coef']:.4f}, SE={bb['se']:.4f}, p={bb['pval']:.4g}, 95%CI=[{bb['ci_lower']:.4f}, {bb['ci_upper']:.4f}].")
    else:
        desc_lines.append("BeautyBinary not included in the model output.")

    if "BeautyIndex_x_Male" in results_by_variable:
        bim = results_by_variable["BeautyIndex_x_Male"]
        desc_lines.append(f"Interaction BeautyIndex x Male: coef={bim['coef']:.4f}, SE={bim['se']:.4f}, p={bim['pval']:.4g}, 95%CI=[{bim['ci_lower']:.4f}, {bim['ci_upper']:.4f}].")
    else:
        desc_lines.append("No BeautyIndex x Male interaction coefficient reported in the model output.")

    # Marginal effects summary
    for name, me in marginal_effects.items():
        desc_lines.append(f"{name}: effect={me['est']:.4f}, SE={me['se']:.4f}, p={me['pval']:.4g}, 95%CI=[{me['ci_lower']:.4f}, {me['ci_upper']:.4f}].")

    # Final interpretive hint
    desc_lines.append("Interpretation: coefficients reflect change in EvalScore (1-5 scale) per unit increase in the predictor (or difference for binary). Statistical significance indicates evidence that beauty is associated with evaluation scores after controlling for included covariates; marginal effects for males/females account for the interaction term if present.")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}