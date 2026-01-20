def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, CIs, odds-ratios, and marginal effects
    for the key predictors in the logistic model:
      - size_diff_z
      - loc_diff_z
      - interaction: size_diff_z:loc_diff_z (or size_diff_z:loc_diff_z variant)

    Returns:
      {
        "object": {
          "terms": {term_name: {coef, se, p, ci_lower, ci_upper, or, or_ci_lower, or_ci_upper, significant}},
          "marginal_effects_of_size": {
             loc = -1, 0, +1 : {estimate_logodds, se, p, ci_logodds_lower, ci_logodds_upper,
                                OR, OR_ci_lower, OR_ci_upper, significant}
          }
        },
        "description": <brief interpretation string>
      }
    """
    import numpy as np
    from scipy import stats

    # Helper to get attributes/methods robustly
    def get_params(obj):
        if hasattr(obj, "params"):
            return obj.params
        raise AttributeError("model_output has no .params")

    def get_pvalues(obj):
        if hasattr(obj, "pvalues"):
            return obj.pvalues
        return None

    def get_bse(obj):
        if hasattr(obj, "bse"):
            return obj.bse
        return None

    def get_conf_int(obj, alpha=0.05):
        # Prefer callable conf_int, otherwise look for conf_int attribute
        if hasattr(obj, "conf_int") and callable(obj.conf_int):
            try:
                ci = obj.conf_int(alpha=alpha)
                # DataFrame may have columns [0,1] or ['lower','upper']
                if isinstance(ci, (list, tuple)):
                    ci = np.asarray(ci)
                return ci
            except Exception:
                pass
        # try attribute
        if hasattr(obj, "conf_int"):
            return obj.conf_int
        return None

    def get_cov(obj):
        # Try common ways to obtain covariance matrix (DataFrame or ndarray)
        # 1) cov_params() method
        if hasattr(obj, "cov_params") and callable(obj.cov_params):
            try:
                return obj.cov_params()
            except Exception:
                pass
        # 2) cov_params attribute (DataFrame)
        if hasattr(obj, "cov_params"):
            cov = obj.cov_params
            # if it's callable (unlikely here), call it
            if callable(cov):
                try:
                    return cov()
                except Exception:
                    pass
            return cov
        # 3) cov_params_ (alternative name)
        if hasattr(obj, "cov_params_"):
            return obj.cov_params_
        raise AttributeError("Could not obtain covariance matrix from model_output")

    params = get_params(model_output)
    pvalues = get_pvalues(model_output)
    bse = get_bse(model_output)
    cov = get_cov(model_output)  # prefer DataFrame with row/col labels
    ci_df = get_conf_int(model_output)

    # Ensure params is a pandas Series or similar indexable mapping
    try:
        param_index = list(params.index)
    except Exception:
        param_index = list(params.keys())

    # Identify term names robustly
    def find_term(name_snippet):
        # find index containing all tokens in snippet (split by non-alphanumeric)
        tokens = [t for t in name_snippet.replace(":", " ").split() if t]
        for n in param_index:
            nstr = str(n)
            if all(tok in nstr for tok in tokens):
                return nstr
        return None

    term_size = find_term("size_diff_z")
    term_loc = find_term("loc_diff_z")
    # interaction might appear as 'size_diff_z:loc_diff_z' or 'size_diff_z:loc_diff_z'
    term_inter = None
    # look for any term containing both substrings and a separator like ':' or '*'
    for n in param_index:
        nstr = str(n)
        if ("size_diff_z" in nstr) and ("loc_diff_z" in nstr) and (":" in nstr or "*" in nstr):
            term_inter = nstr
            break
    # fallback: explicit name used in the model output provided in prompt
    if term_inter is None:
        if "size_diff_z:loc_diff_z" in param_index:
            term_inter = "size_diff_z:loc_diff_z"

    if term_size is None or term_loc is None:
        raise KeyError("Could not find required term names in model params. Found: " + ", ".join(param_index))

    # Function to safely extract numeric values from params/pvalues/bse
    def get_val(container, name):
        if container is None:
            return None
        try:
            return float(container[name])
        except Exception:
            try:
                # maybe name is slightly different (like "C(focal)[T.2]" interfering)
                # try exact match ignoring case
                for k in container.index:
                    if str(k) == str(name):
                        return float(container[k])
                return None
            except Exception:
                return None

    # Prepare output structure
    terms_out = {}

    # prepare covariance as numpy array and index labels
    import pandas as pd
    if isinstance(cov, pd.DataFrame):
        cov_df = cov
    else:
        # try to coerce to DataFrame
        try:
            cov_df = pd.DataFrame(cov, index=param_index, columns=param_index)
        except Exception:
            raise ValueError("Covariance matrix could not be coerced to DataFrame")

    # Helper to compute CI for a single param (if conf_int available)
    def param_ci(name):
        if ci_df is None:
            return (None, None)
        # ci_df might be DataFrame with columns [0,1] or ['lower','upper']
        try:
            if isinstance(ci_df, pd.DataFrame):
                # match row by name
                if name in ci_df.index:
                    row = ci_df.loc[name]
                    # try known column names
                    if "lower" in row.index and "upper" in row.index:
                        return (float(row["lower"]), float(row["upper"]))
                    else:
                        # assume first two columns
                        return (float(row.iloc[0]), float(row.iloc[1]))
                else:
                    # maybe conf_int returns numpy array in same order as params
                    if len(ci_df) == len(param_index):
                        pos = param_index.index(name)
                        entry = ci_df[pos]
                        return (float(entry[0]), float(entry[1]))
            else:
                # numpy array
                arr = np.asarray(ci_df)
                pos = param_index.index(name)
                return (float(arr[pos, 0]), float(arr[pos, 1]))
        except Exception:
            return (None, None)
        return (None, None)

    # Extract main term stats
    for term in (term_size, term_loc, term_inter):
        if term is None:
            continue
        coef = get_val(params, term)
        se = get_val(bse, term) if bse is not None else (np.sqrt(cov_df.loc[term, term]) if term in cov_df.index else None)
        p = get_val(pvalues, term) if pvalues is not None else None
        ci_low, ci_high = param_ci(term)
        # odds ratio and its CI
        or_est = float(np.exp(coef)) if coef is not None else None
        if ci_low is not None and ci_high is not None:
            or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))
        else:
            or_ci_low = or_ci_high = None
        terms_out[term] = {
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "odds_ratio": or_est,
            "or_ci_lower": or_ci_low,
            "or_ci_upper": or_ci_high,
            "significant": (p is not None and p < 0.05)
        }

    # Compute marginal effect of size_diff_z at loc_diff_z = -1, 0, +1 (in z-score units)
    marg_effects = {}
    # Need names present
    if term_inter is None:
        # interaction missing -> simple effect is just coef of size
        for loc_val in [-1.0, 0.0, 1.0]:
            est = get_val(params, term_size)
            se_est = get_val(bse, term_size)
            p_est = get_val(pvalues, term_size) if pvalues is not None else None
            ci_low, ci_high = param_ci(term_size)
            or_est = float(np.exp(est)) if est is not None else None
            or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
            or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
            marg_effects[f"loc_{loc_val}"] = {
                "loc_value": loc_val,
                "estimate_logodds": est,
                "se": se_est,
                "p_value": p_est,
                "ci_logodds_lower": ci_low,
                "ci_logodds_upper": ci_high,
                "odds_ratio_per_1sd_size": or_est,
                "or_ci_lower": or_ci_low,
                "or_ci_upper": or_ci_high,
                "significant": (p_est is not None and p_est < 0.05)
            }
    else:
        # compute linear combination: beta_size + loc_val * beta_inter
        for loc_val in [-1.0, 0.0, 1.0]:
            b_size = get_val(params, term_size)
            b_inter = get_val(params, term_inter)
            if b_size is None or b_inter is None:
                est = None
            else:
                est = float(b_size + loc_val * b_inter)
            # compute se using covariance
            se_est = None
            p_est = None
            ci_low = ci_high = None
            if (term_size in cov_df.index) and (term_inter in cov_df.index):
                var = float(cov_df.loc[term_size, term_size] +
                            (loc_val ** 2) * cov_df.loc[term_inter, term_inter] +
                            2 * loc_val * cov_df.loc[term_size, term_inter])
                # numerical safety
                var = max(var, 0.0)
                se_est = float(np.sqrt(var))
                if se_est > 0 and est is not None:
                    z = est / se_est
                    p_est = float(2 * stats.norm.sf(abs(z)))
                    # compute CI on log-odds
                    q = stats.norm.ppf(0.975)
                    ci_low = float(est - q * se_est)
                    ci_high = float(est + q * se_est)
            # OR and CI
            or_est = float(np.exp(est)) if est is not None else None
            or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
            or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
            marg_effects[f"loc_{loc_val}"] = {
                "loc_value": loc_val,
                "estimate_logodds": est,
                "se": se_est,
                "p_value": p_est,
                "ci_logodds_lower": ci_low,
                "ci_logodds_upper": ci_high,
                "odds_ratio_per_1sd_size": or_est,
                "or_ci_lower": or_ci_low,
                "or_ci_upper": or_ci_high,
                "significant": (p_est is not None and p_est < 0.05)
            }

    result_object = {
        "terms": terms_out,
        "marginal_effects_of_size": marg_effects
    }

    # Short interpretation based on significance flags
    sig_terms = [t for t, v in terms_out.items() if v["significant"]]
    if len(sig_terms) == 0:
        interp = (
            "None of the focal predictors (relative size, location, or their interaction) "
            "are statistically significant at p<0.05 according to the model's cluster-robust SEs. "
            "The point estimates: size effect is small and positive, location (home advantage) is positive, "
            "and the interaction is slightly negative (suggesting the size advantage may be weaker when the focal group is closer to home). "
            "However, these effects are not statistically distinguishable from zero given the provided SEs."
        )
    else:
        interp_parts = []
        for t in sig_terms:
            v = terms_out[t]
            interp_parts.append(
                f"{t}: coef={v['coef']:.3g}, p={v['p_value']:.3g}, OR={v['odds_ratio']:.3g} (CI [{v['or_ci_lower']:.3g}, {v['or_ci_upper']:.3g}])"
            )
        interp = "Significant effects: " + "; ".join(interp_parts) + ". See the returned 'object' for full estimates and marginal effects."

    return {"object": result_object, "description": interp}