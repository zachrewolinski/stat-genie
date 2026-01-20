import math
import numpy as np
import pandas as pd

def extract_final_answer(model_output):
    """
    Extracts coefficients, robust standard errors, p-values, confidence intervals,
    and computes the marginal effect of beauty at the centered mean (beauty=0)
    for male (female=0) and female (female=1) instructors.

    Returns:
      {
        "object": {
          "coefficients": {
            "beauty": {"coef": ..., "se": ..., "p": ..., "ci_2.5%": ..., "ci_97.5%": ...},
            "beauty_sq": {...},
            "beauty:female": {...}
          },
          "marginal_effects_at_mean": {
            "male": {"effect": ..., "se": ..., "ci_2.5%": ..., "ci_97.5%": ..., "p": ...},
            "female": {"effect": ..., "se": ..., "ci_2.5%": ..., "ci_97.5%": ..., "p": ...}
          }
        },
        "description": "Brief interpretation string"
      }
    """
    # Helper: normal two-sided p-value from z
    def two_sided_p_from_z(z):
        # z is non-negative (we call with abs(z) typically), but handle any sign
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return float(2.0 * (1.0 - cdf)) if z >= 0 else float(2.0 * cdf)

    # Retrieve raw params, pvalues, cov, conf robustly and normalize to pandas objects
    raw_params = getattr(model_output, "params", None)
    # Determine parameter names
    param_names = None
    if isinstance(raw_params, pd.Series):
        params = raw_params.astype(float)
        param_names = list(params.index)
    elif isinstance(raw_params, np.ndarray):
        # Try several places to get parameter names
        for attr in ("param_names", "params_names", "exog_names", "k_names", "names"):
            if hasattr(model_output, attr):
                val = getattr(model_output, attr)
                if isinstance(val, (list, tuple, np.ndarray)):
                    param_names = list(val)
                    break
        if param_names is None and hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            param_names = list(model_output.model.exog_names)
        if param_names is None:
            # Fallback to generic names
            param_names = [f"b{i}" for i in range(len(raw_params))]
        params = pd.Series(np.asarray(raw_params, dtype=float), index=param_names)
    elif isinstance(raw_params, dict):
        params = pd.Series(raw_params, dtype=float)
        param_names = list(params.index)
    else:
        # Try to coerce to Series
        try:
            params = pd.Series(model_output.params)
            params = params.astype(float)
            param_names = list(params.index)
        except Exception:
            raise ValueError("Unable to interpret model_output.params. Expected Series, ndarray, or dict.")

    # cov_params
    raw_cov = None
    if hasattr(model_output, "cov_params"):
        try:
            raw_cov = model_output.cov_params()
        except Exception:
            raw_cov = None
    if isinstance(raw_cov, pd.DataFrame):
        cov = raw_cov.astype(float)
    elif isinstance(raw_cov, np.ndarray):
        # ensure shape matches param_names
        cov = pd.DataFrame(raw_cov.astype(float), index=param_names, columns=param_names)
    elif raw_cov is None:
        # try attribute
        raw_cov = getattr(model_output, "cov", None)
        if isinstance(raw_cov, (pd.DataFrame, np.ndarray)):
            if isinstance(raw_cov, pd.DataFrame):
                cov = raw_cov.astype(float)
            else:
                cov = pd.DataFrame(np.asarray(raw_cov, dtype=float), index=param_names, columns=param_names)
        else:
            # fallback: zero cov (should not happen for real model_output)
            cov = pd.DataFrame(np.zeros((len(param_names), len(param_names))), index=param_names, columns=param_names)
    else:
        # try to coerce
        try:
            cov = pd.DataFrame(raw_cov, index=param_names, columns=param_names).astype(float)
        except Exception:
            cov = pd.DataFrame(np.zeros((len(param_names), len(param_names))), index=param_names, columns=param_names)

    # pvalues
    raw_pvalues = getattr(model_output, "pvalues", None)
    if isinstance(raw_pvalues, pd.Series):
        pvalues = raw_pvalues.astype(float)
    elif isinstance(raw_pvalues, np.ndarray):
        pvalues = pd.Series(np.asarray(raw_pvalues, dtype=float), index=param_names)
    elif isinstance(raw_pvalues, dict):
        pvalues = pd.Series(raw_pvalues, dtype=float)
    else:
        try:
            pvalues = pd.Series(model_output.pvalues).astype(float)
        except Exception:
            # fallback: NaNs
            pvalues = pd.Series({n: float("nan") for n in param_names})

    # conf_int: expect shape (k,2) or DataFrame
    raw_conf = None
    if hasattr(model_output, "conf_int"):
        try:
            raw_conf = model_output.conf_int()
        except Exception:
            raw_conf = None
    if isinstance(raw_conf, pd.DataFrame):
        conf = raw_conf.copy()
    elif isinstance(raw_conf, np.ndarray):
        conf = pd.DataFrame(raw_conf, index=param_names, columns=[0,1]).astype(float)
    elif raw_conf is None:
        # fallback: use coef +/- 1.96*se if se available from cov
        cis = []
        for name in param_names:
            coef = float(params[name])
            se = float(math.sqrt(float(cov.loc[name, name]))) if name in cov.index else float("nan")
            cis.append((coef - 1.96 * se, coef + 1.96 * se))
        conf = pd.DataFrame(cis, index=param_names, columns=[0,1]).astype(float)
    else:
        try:
            conf = pd.DataFrame(raw_conf)
        except Exception:
            conf = pd.DataFrame([[float("nan"), float("nan")] for _ in param_names], index=param_names, columns=[0,1])

    # Helper to find parameter name in param_names robustly
    def get_param_name(base):
        # Exact match
        if base in params.index:
            return base
        # Common alternative separators
        alternatives = [
            base,
            base.replace(":", "__"),
            base.replace(":", "."),
            base.replace(":", "_"),
            base.replace(":", "/"),
        ]
        for alt in alternatives:
            if alt in params.index:
                return alt
        # Try matching tokens (order-insensitive for interaction)
        if ":" in base:
            toks = base.split(":")
            for name in params.index:
                if all(tok in name for tok in toks):
                    return name
        # Try substring match for simple names
        base_root = base.split(":")[0]
        for name in params.index:
            if base_root == name or base_root in name:
                return name
        return None

    name_beauty = get_param_name("beauty")
    name_beauty_sq = get_param_name("beauty_sq")
    name_inter = get_param_name("beauty:female")

    missing = [n for n in (name_beauty, name_beauty_sq, name_inter) if n is None]
    if missing:
        raise ValueError(
            "Model output does not contain expected parameter(s): "
            "beauty, beauty_sq, beauty:female. Found parameters: "
            f"{list(params.index)}"
        )

    # Extract stats for each param
    def gather_stats(param_name):
        coef = float(params[param_name])
        se = float(np.sqrt(float(cov.loc[param_name, param_name]))) if param_name in cov.index else float("nan")
        p = float(pvalues[param_name]) if param_name in pvalues.index else float("nan")
        ci_low = float(conf.loc[param_name, 0]) if param_name in conf.index else float("nan")
        ci_high = float(conf.loc[param_name, 1]) if param_name in conf.index else float("nan")
        return {"coef": coef, "se": se, "p": p, "ci_2.5%": ci_low, "ci_97.5%": ci_high}

    stats_beauty = gather_stats(name_beauty)
    stats_beauty_sq = gather_stats(name_beauty_sq)
    stats_inter = gather_stats(name_inter)

    # Marginal effect of beauty at beauty = 0 (centered mean)
    beta_b = stats_beauty["coef"]
    beta_bsq = stats_beauty_sq["coef"]
    beta_int = stats_inter["coef"]

    # Variances and covariances needed for SE of linear combinations
    var_b = float(cov.loc[name_beauty, name_beauty])
    var_int = float(cov.loc[name_inter, name_inter])
    cov_b_int = float(cov.loc[name_beauty, name_inter]) if (name_beauty in cov.index and name_inter in cov.columns) else 0.0
    # beauty_sq does not enter marginal at centered value 0

    # Male
    eff_male = float(beta_b)
    se_male = float(math.sqrt(var_b)) if var_b >= 0 else float('nan')
    z_male = eff_male / se_male if se_male > 0 else float('nan')
    p_male = two_sided_p_from_z(abs(z_male)) if not math.isnan(z_male) else float('nan')
    ci_male = (eff_male - 1.96 * se_male, eff_male + 1.96 * se_male) if not math.isnan(se_male) else (float('nan'), float('nan'))

    # Female
    eff_female = float(beta_b + beta_int)
    var_female = var_b + var_int + 2.0 * cov_b_int
    se_female = float(math.sqrt(var_female)) if var_female >= 0 else float('nan')
    z_female = eff_female / se_female if se_female > 0 else float('nan')
    p_female = two_sided_p_from_z(abs(z_female)) if not math.isnan(z_female) else float('nan')
    ci_female = (eff_female - 1.96 * se_female, eff_female + 1.96 * se_female) if not math.isnan(se_female) else (float('nan'), float('nan'))

    result_object = {
        "coefficients": {
            "beauty": stats_beauty,
            "beauty_sq": stats_beauty_sq,
            "beauty:female": stats_inter
        },
        "marginal_effects_at_mean": {
            "male": {
                "effect": eff_male,
                "se": se_male,
                "ci_2.5%": float(ci_male[0]),
                "ci_97.5%": float(ci_male[1]),
                "p": p_male
            },
            "female": {
                "effect": eff_female,
                "se": se_female,
                "ci_2.5%": float(ci_female[0]),
                "ci_97.5%": float(ci_female[1]),
                "p": p_female
            }
        }
    }

    # Short interpretation
    sig_male = (p_male < 0.05) if (not math.isnan(p_male)) else False
    sig_female = (p_female < 0.05) if (not math.isnan(p_female)) else False
    sig_bsq = (stats_beauty_sq["p"] < 0.05) if (not math.isnan(stats_beauty_sq["p"])) else False

    desc_lines = []
    desc_lines.append(
        f"At the centered mean of beauty (beauty=0): the marginal effect of beauty on "
        f"evaluation for male instructors is {eff_male:.4f} (SE={se_male:.4f}, p={p_male:.3f})."
        + (" Statistically significant." if sig_male else " Not statistically significant.")
    )
    desc_lines.append(
        f"For female instructors the marginal effect at mean is {eff_female:.4f} (SE={se_female:.4f}, p={p_female:.3f})."
        + (" Statistically significant." if sig_female else " Not statistically significant.")
    )
    if sig_bsq:
        desc_lines.append(
            f"The quadratic term (beauty_sq) is statistically significant (p={stats_beauty_sq['p']:.3f}), "
            "indicating a nonlinear relationship between beauty and evaluations."
        )
    else:
        desc_lines.append(
            f"The quadratic term (beauty_sq) is not statistically significant (p={stats_beauty_sq['p']:.3f}), "
            "so there is no strong evidence of nonlinearity in beauty's effect at conventional levels."
        )

    desc_lines.append(
        "All reported standard errors, p-values, and confidence intervals are those from the covariance matrix "
        "returned with the model_output (converted to a DataFrame for calculations)."
    )

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}