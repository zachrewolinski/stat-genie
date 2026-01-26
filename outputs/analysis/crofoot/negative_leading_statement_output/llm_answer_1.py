def extract_final_answer(model_output):
    """
    Extracts coefficients, odds ratios, 95% CIs, p-values, and a short interpretation
    for the focal effects (SizeAdv_s, LocationAdv_s) and their interaction from the
    fitted GLM model contained in model_output.

    Returns:
      {
        "object": {
          "n_obs": int,
          "aic": float,
          "variables": {
            varname: {
              "coef": float,
              "SE": float,
              "OR": float,
              "CI_coef": (float, float),
              "CI_OR": (float, float),
              "p_value": float
            }, ...
          },
          "marginal_effects_of_size_at_location": {
            loc_value: { "effect_coef": float, "SE": float, "OR": float, "p_value": float }, ...
          }  # only included if interaction term present
        },
        "description": "string summary interpretation"
      }
    """
    import numpy as np
    import pandas as pd
    from math import isfinite
    from scipy.stats import norm

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the modeling function.")
    if 'model' not in model_output:
        raise ValueError("model_output must contain key 'model' with the fitted statsmodels object.")

    fit = model_output['model']

    # Extract basic fit pieces
    params = fit.params
    pvalues = fit.pvalues
    try:
        conf = fit.conf_int()  # DataFrame or array-like with two columns
    except Exception:
        conf = None
    try:
        bse = fit.bse
    except Exception:
        # fallback: approximate SE from confint if available
        if conf is not None:
            bse = (conf.iloc[:, 1] - conf.iloc[:, 0]) / (2 * 1.96)
        else:
            raise

    result_obj = {
        "n_obs": int(model_output.get('n_obs', getattr(fit, 'nobs', np.nan))),
        "aic": model_output.get('aic', getattr(fit, 'aic', np.nan)),
        "variables": {}
    }

    # Variables of interest
    focal_vars = ['SizeAdv_s', 'LocationAdv_s', 'SizeAdv_s:LocationAdv_s']
    for var in focal_vars:
        if var in params.index:
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else float(np.nan)
            p = float(pvalues[var]) if var in pvalues.index else float(np.nan)
            if conf is not None and var in conf.index:
                ci_low, ci_high = float(conf.loc[var, 0]), float(conf.loc[var, 1])
            else:
                ci_low, ci_high = (coef - 1.96 * se, coef + 1.96 * se) if not np.isnan(se) else (np.nan, np.nan)
            # Odds ratio and its CI
            or_val = float(np.exp(coef)) if isfinite(coef) else float('nan')
            or_ci_low = float(np.exp(ci_low)) if isfinite(ci_low) else float('nan')
            or_ci_high = float(np.exp(ci_high)) if isfinite(ci_high) else float('nan')

            result_obj["variables"][var] = {
                "coef": coef,
                "SE": se,
                "OR": or_val,
                "CI_coef": (ci_low, ci_high),
                "CI_OR": (or_ci_low, or_ci_high),
                "p_value": p
            }

    # If interaction present, compute marginal effect of SizeAdv_s at LocationAdv_s = -1, 0, +1 (std units)
    inter_name = 'SizeAdv_s:LocationAdv_s'
    if ('SizeAdv_s' in params.index) and (inter_name in params.index):
        cov = fit.cov_params()
        marg_effects = {}
        for loc in [-1.0, 0.0, 1.0]:
            est = params['SizeAdv_s'] + loc * params[inter_name]
            # standard error of linear combination
            var_size = cov.loc['SizeAdv_s', 'SizeAdv_s']
            var_inter = cov.loc[inter_name, inter_name]
            covar = cov.loc['SizeAdv_s', inter_name]
            se_lin = np.sqrt(var_size + (loc ** 2) * var_inter + 2 * loc * covar)
            z = est / se_lin if se_lin > 0 else np.nan
            p_lin = float(2 * (1 - norm.cdf(abs(z)))) if isfinite(z) else float(np.nan)
            marg_effects[loc] = {
                "effect_coef": float(est),
                "SE": float(se_lin),
                "OR": float(np.exp(est)) if isfinite(est) else float('nan'),
                "p_value": p_lin
            }
        result_obj["marginal_effects_of_size_at_location"] = marg_effects

    # Build description string summarizing main findings (statistical significance at alpha=0.05)
    lines = []
    n = result_obj["n_obs"]
    lines.append(f"Sample size: {n} contests.")
    for var, stats in result_obj["variables"].items():
        coef = stats["coef"]
        OR = stats["OR"]
        ci_or = stats["CI_OR"]
        p = stats["p_value"]
        sig = "statistically significant" if (isinstance(p, float) and (p < 0.05)) else "not statistically significant"
        direction = ""
        if coef > 0:
            direction = "positive"
        elif coef < 0:
            direction = "negative"
        else:
            direction = "no clear"
        lines.append(
            f"{var}: coef={coef:.3f}, OR={OR:.3f}, 95% CI for OR=({ci_or[0]:.3f}, {ci_or[1]:.3f}), p={p:.3g} -> {sig} ({direction} effect)."
        )

    # Interpret interaction explicitly
    if inter_name in result_obj["variables"]:
        p_int = result_obj["variables"][inter_name]["p_value"]
        if p_int < 0.05:
            lines.append(
                "The interaction between SizeAdv and LocationAdv is statistically significant, "
                "meaning the effect of relative group size on winning depends on contest location. "
                "Marginal effects of size at representative LocationAdv values are provided."
            )
        else:
            lines.append(
                "The interaction between SizeAdv and LocationAdv is not statistically significant, "
                "so there is no strong evidence that the effect of relative group size depends on contest location."
            )
        # include marginal effects summary
        me = result_obj.get("marginal_effects_of_size_at_location", {})
        for loc, v in me.items():
            pv = v["p_value"]
            sig = "significant" if (isinstance(pv, float) and (pv < 0.05)) else "not significant"
            lines.append(
                f"Marginal effect of SizeAdv at LocationAdv={loc:+.1f}: coef={v['effect_coef']:.3f}, OR={v['OR']:.3f}, p={pv:.3g} ({sig})."
            )

    description = " ".join(lines)

    return {"object": result_obj, "description": description}