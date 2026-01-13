def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderViewOn (and its interaction with Dyslexia categories)
    from a fitted statsmodels OLSResults-like object and returns summary statistics for each dyslexia group.

    Returns:
      {
        "object": {
           "<group_label>": {
               "coef": float,            # estimated effect (words/second) of turning ReaderViewOn for this group
               "se": float,              # standard error of that estimated effect
               "t": float,               # t-statistic
               "p_value": float,         # two-sided p-value
               "ci_lower": float,        # lower bound of 95% CI
               "ci_upper": float         # upper bound of 95% CI
           },
           ... (one entry per group: "No dyslexia", "Dyslexia", "Severe dyslexia")
        },
        "description": str  # brief interpretation of the numbers and whether ReaderView improves reading speed for each group
      }
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Safe extraction of parameter values and names
    raw_params = getattr(model_output, "params", None)
    if raw_params is None:
        raise ValueError("model_output has no 'params' attribute.")

    # Get parameter values as numpy array
    params_values = np.asarray(raw_params)

    # Try to get parameter names from various possible locations
    param_names = None
    # If params has an index (pandas Series), use it
    if hasattr(raw_params, "index"):
        try:
            param_names = list(raw_params.index)
        except Exception:
            param_names = None

    # Try model.exog_names (statsmodels)
    if param_names is None and hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
        try:
            param_names = list(model_output.model.exog_names)
        except Exception:
            param_names = None

    # Try attribute 'param_names' if present
    if param_names is None and hasattr(model_output, "param_names"):
        try:
            param_names = list(model_output.param_names)
        except Exception:
            param_names = None

    # Fallback: auto-generate names
    if param_names is None:
        param_names = [f"param_{i}" for i in range(len(params_values))]

    # Build a pandas Series for params for convenient name-based access
    params = pd.Series(params_values, index=param_names)

    # Covariance: handle both DataFrame and ndarray
    cov_raw = None
    if hasattr(model_output, "cov_params"):
        try:
            cov_raw = model_output.cov_params()
        except Exception:
            cov_raw = None

    if cov_raw is None:
        # fallback: try to access attribute directly
        cov_raw = getattr(model_output, "cov_params_", None)

    if cov_raw is None:
        # If no covariance available, create zeros
        cov = pd.DataFrame(np.zeros((len(param_names), len(param_names))), index=param_names, columns=param_names)
    else:
        if isinstance(cov_raw, pd.DataFrame):
            cov = cov_raw.reindex(index=param_names, columns=param_names).fillna(0.0)
        elif isinstance(cov_raw, np.ndarray):
            cov = pd.DataFrame(cov_raw, index=param_names, columns=param_names)
        else:
            # Try to convert to ndarray and then DataFrame
            try:
                carr = np.asarray(cov_raw)
                cov = pd.DataFrame(carr, index=param_names, columns=param_names)
            except Exception:
                cov = pd.DataFrame(np.zeros((len(param_names), len(param_names))), index=param_names, columns=param_names)

    # Residual degrees of freedom
    df_resid = None
    if hasattr(model_output, "df_resid"):
        try:
            df_resid = float(model_output.df_resid)
        except Exception:
            df_resid = None

    # Helper: safe get of parameter by exact name
    def get_param_value(name):
        return float(params[name]) if name in params.index else 0.0

    # Helper: safe covariance lookup
    def safe_cov(a, b):
        if a in cov.index and b in cov.columns:
            return float(cov.at[a, b])
        if b in cov.index and a in cov.columns:
            return float(cov.at[b, a])
        return 0.0

    # Base effect of ReaderViewOn (for the reference category of Dyslexia_cat)
    if 'ReaderViewOn' in params.index:
        beta_rv = float(params['ReaderViewOn'])
        rv_name = 'ReaderViewOn'
    else:
        # Try to find the parameter name that corresponds to ReaderViewOn (tolerant)
        rv_matches = [n for n in params.index if re.search(r'\bReaderViewOn\b', n)]
        if rv_matches:
            rv_name = rv_matches[0]
            beta_rv = float(params[rv_name])
        else:
            raise ValueError("ReaderViewOn main effect not found among model parameters.")

    # Find interaction terms of ReaderViewOn with Dyslexia categories
    interaction_terms = {}  # mapping level -> param_name
    for name in params.index:
        if 'ReaderViewOn' in name and 'C(Dyslexia_cat)' in name:
            # name examples:
            # 'ReaderViewOn:C(Dyslexia_cat)[T.Dyslexia]'
            # 'C(Dyslexia_cat)[T.Dyslexia]:ReaderViewOn'
            m = re.search(r'C\(Dyslexia_cat\)\[T\.(.+?)\]', name)
            if m:
                level = m.group(1)
            else:
                # fallback: try to extract text after T.
                m2 = re.search(r'\[T\.(.+?)\]', name)
                level = m2.group(1) if m2 else name
            interaction_terms[level] = name
        else:
            # Also handle cases where the interaction is represented as 'C(Dyslexia_cat)[T.X]:ReaderViewOn'
            if 'C(Dyslexia_cat)' in name and ':' in name and 'ReaderViewOn' in name:
                m = re.search(r'C\(Dyslexia_cat\)\[T\.(.+?)\]', name)
                if m:
                    level = m.group(1)
                else:
                    m2 = re.search(r'\[T\.(.+?)\]', name)
                    level = m2.group(1) if m2 else name
                interaction_terms[level] = name

    # We'll report for the expected category labels (as provided in the task).
    target_levels = ["No dyslexia", "Dyslexia", "Severe dyslexia"]

    results = {}
    t_crit = None
    if df_resid is not None and df_resid > 0:
        t_crit = stats.t.ppf(1 - 0.025, df_resid)

    for lvl in target_levels:
        # For reference category ("No dyslexia") the effect is just beta_rv
        if lvl == "No dyslexia":
            coef = beta_rv
            var = safe_cov(rv_name, rv_name)
        else:
            # find the interaction parameter for this level (the name stored in interaction_terms uses the exact level string)
            interaction_name = None
            if lvl in interaction_terms:
                interaction_name = interaction_terms[lvl]
            else:
                # try to find an interaction whose extracted level contains tokens from lvl
                lvl_tokens = [t.lower() for t in re.split(r'\s+', lvl) if t]
                for extracted_level, pname in interaction_terms.items():
                    elow = extracted_level.lower()
                    if all(tok in elow for tok in lvl_tokens):
                        interaction_name = pname
                        break
            if interaction_name is None:
                # No interaction term found for this level: effect equals base effect
                coef = beta_rv
                var = safe_cov(rv_name, rv_name)
            else:
                beta_int = float(params[interaction_name]) if interaction_name in params.index else 0.0
                coef = beta_rv + beta_int
                # variance of sum: Var(a+b) = Var(a)+Var(b)+2*Cov(a,b)
                var_a = safe_cov(rv_name, rv_name)
                var_b = safe_cov(interaction_name, interaction_name)
                cov_ab = safe_cov(rv_name, interaction_name)
                var = var_a + var_b + 2.0 * cov_ab

        se = float(np.sqrt(var)) if var >= 0 else float('nan')
        t_stat = float(coef / se) if se != 0 and not np.isnan(se) else float('nan')
        # two-sided p-value using t-distribution if df_resid available, else normal approx
        if df_resid is not None and df_resid > 0 and not np.isnan(t_stat):
            p_value = float(stats.t.sf(abs(t_stat), df_resid) * 2.0)
            if t_crit is not None:
                ci_lower = coef - t_crit * se
                ci_upper = coef + t_crit * se
            else:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
        else:
            # normal approx
            p_value = float(stats.norm.sf(abs(t_stat)) * 2.0) if not np.isnan(t_stat) else float('nan')
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

        results[lvl] = {
            "coef": float(coef),
            "se": float(se),
            "t": float(t_stat),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        }

    # Build a concise interpretation string
    interpretations = []
    for lvl in target_levels:
        r = results[lvl]
        signif = (r["p_value"] < 0.05) if (r["p_value"] is not None and not np.isnan(r["p_value"])) else False
        direction = "increase" if r["coef"] > 0 else ("decrease" if r["coef"] < 0 else "no change")
        p_str = f"{r['p_value']:.3g}" if (r["p_value"] is not None and not np.isnan(r["p_value"])) else "NA"
        interp = (f"For '{lvl}': estimated effect of ReaderViewOn = {r['coef']:.4f} words/sec "
                  f"(SE={r['se']:.4f}, 95% CI [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}], p={p_str}). "
                  f"This corresponds to a {direction} in reading speed; "
                  f"{'statistically significant at alpha=0.05.' if signif else 'not statistically significant.'}")
        interpretations.append(interp)

    description = ("Extracted marginal effects of ReaderViewOn on ReadingSpeed_wps for each dyslexia group (effects are in words/sec). "
                   "Effects are the estimated change in reading speed when Reader View is turned ON compared to OFF, "
                   "computed for the reference group (No dyslexia) and for Dyslexia / Severe dyslexia via the interaction terms. "
                   "Interpretation by group:\n- " + "\n- ".join(interpretations))

    return {"object": results, "description": description}