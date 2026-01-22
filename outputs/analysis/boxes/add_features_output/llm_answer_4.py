def extract_final_answer(model_output):
    """
    Extracts age-related effects (slopes) on choosing the majority option from the
    fitted GLM (binary MajorityChoice) and tests whether those age slopes differ
    across cultures (age x culture interactions).

    Returns a dict with:
      - "object": dict mapping each culture (including the reference/base culture)
                  to the estimated age slope (beta), SE, z, p, 95% CI.
                 Also includes a joint-test p-value for the set of interaction terms
                 (if available) under the key "interaction_joint_test_p".
      - "description": short textual interpretation of the results in context.

    The function is defensive: it tries to use robust covariance information
    from the provided fitted results and falls back to reasonable defaults if
    a particular extraction step fails.
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    out = {
        "object": None,
        "description": None
    }

    # Check we have the GLM result
    try:
        glm_res = model_output.get('glm_majority') if isinstance(model_output, dict) else model_output['glm_majority']
    except Exception:
        glm_res = None

    if glm_res is None:
        raise KeyError("model_output must contain key 'glm_majority' with fitted GLM results.")

    # Extract parameter estimates and (robust) covariance matrix
    try:
        params = glm_res.params.copy()  # pandas Series expected
        # ensure it's a pandas Series
        if not isinstance(params, pd.Series):
            params = pd.Series(params)
    except Exception as e:
        raise RuntimeError(f"Couldn't extract params from glm_majority: {e}")

    # Try to get covariance matrix; coerce to DataFrame indexed by parameter names
    cov = None
    try:
        cov_raw = glm_res.cov_params()
    except Exception:
        # try common alternative attribute names
        cov_raw = None
        for attr in ("normalized_cov_params", "cov_params_default", "bse_cov"):
            cov_raw = getattr(glm_res, attr, None)
            if cov_raw is not None:
                break
    if cov_raw is None:
        raise RuntimeError("Couldn't extract covariance matrix from glm_majority result.")

    # Coerce cov_raw into a pandas DataFrame with proper index/columns corresponding to params
    try:
        if isinstance(cov_raw, pd.DataFrame):
            cov = cov_raw.copy()
        else:
            # assume array-like
            cov = pd.DataFrame(np.asarray(cov_raw), index=params.index, columns=params.index)
    except Exception:
        # final fallback: create diagonal from parameter standard errors if available
        try:
            bse = glm_res.bse
            if isinstance(bse, (pd.Series, dict, list, np.ndarray)):
                bse_s = pd.Series(bse, index=params.index) if not isinstance(bse, pd.Series) else bse
                cov = pd.DataFrame(np.zeros((len(params), len(params))), index=params.index, columns=params.index)
                for i, nm in enumerate(params.index):
                    cov.loc[nm, nm] = float(bse_s.loc[nm]) ** 2
            else:
                raise RuntimeError("No usable covariance or bse found.")
        except Exception as e:
            raise RuntimeError(f"Couldn't build covariance matrix from glm_majority result: {e}")

    # Identify baseline age coefficient name and interaction names
    param_index = list(params.index)

    # Find exact baseline age term
    baseline_age_name = None
    for n in param_index:
        if n == 'age_centered':
            baseline_age_name = n
            break
    if baseline_age_name is None:
        # try more flexibly: a name that contains 'age_centered'
        for n in param_index:
            if re.search(r'(^|[:\[\]\W])age_centered($|[:\[\]\W])', n):
                baseline_age_name = n
                break
    if baseline_age_name is None:
        # final fallback: any param that contains 'age' and 'center'
        for n in param_index:
            if 'age' in n and 'center' in n:
                baseline_age_name = n
                break
    if baseline_age_name is None:
        raise RuntimeError("Could not find a baseline 'age_centered' parameter in model parameters.")

    # Find culture main effect names and interaction names
    culture_main = []      # entries like 'C(culture)[T.2]'
    interaction_names = [] # entries like 'age_centered:C(culture)[T.2]'
    for n in param_index:
        if 'C(culture)' in n and ':' not in n:
            culture_main.append(n)
        if 'C(culture)' in n and 'age_centered' in n:
            interaction_names.append(n)

    # Extract culture labels from names
    culture_labels = []
    culture_label_map = {}  # map from main effect name -> label
    for name in culture_main:
        m = re.search(r"C\(culture\)\[T\.?([^\]]+)\]$", name)
        label = m.group(1) if m else name
        culture_labels.append(label)
        culture_label_map[name] = label

    # The baseline (reference) culture is the one not listed in dummies.
    all_cultures_report = ['REFERENCE'] + culture_labels

    # For each culture (REFERENCE and each labeled dummy), compute combined age slope:
    results = {}
    try:
        beta_age = float(params[baseline_age_name])
    except Exception:
        beta_age = float(np.asarray(params.loc[baseline_age_name]))
    try:
        var_age = float(cov.loc[baseline_age_name, baseline_age_name])
    except Exception:
        var_age = float(np.asarray(cov.diagonal()[param_index.index(baseline_age_name)])) if hasattr(cov, "values") else np.nan

    # helper to get covariance between two params (0 if missing)
    def cov_param(a, b):
        try:
            return float(cov.loc[a, b])
        except Exception:
            return 0.0

    # Reference (baseline) result
    se_ref = np.sqrt(var_age) if (var_age is not None and var_age >= 0) else np.nan
    z_ref = beta_age / se_ref if (not np.isnan(se_ref) and se_ref != 0) else np.nan
    p_ref = 2 * (1 - stats.norm.cdf(abs(z_ref))) if not np.isnan(z_ref) else np.nan
    ci_ref = (beta_age - 1.96 * se_ref, beta_age + 1.96 * se_ref) if (not np.isnan(se_ref)) else (np.nan, np.nan)
    results['REFERENCE'] = {
        "slope": beta_age,
        "se": se_ref,
        "z": z_ref,
        "p_value": p_ref,
        "ci_95": ci_ref
    }

    # For each culture dummy, find its corresponding interaction param (if any)
    for main_name, label in culture_label_map.items():
        # look for interaction param that contains this label
        interact_name = None
        for iname in interaction_names:
            # accept patterns where the label appears inside the bracket or at end
            if re.search(rf"\[T\.?{re.escape(label)}\]$", iname) and 'age_centered' in iname:
                interact_name = iname
                break
            # also accept order reversed like 'C(culture)[T.x]:age_centered'
            if re.search(rf"C\(culture\)\[T\.?{re.escape(label)}\]", iname) and 'age_centered' in iname:
                interact_name = iname
                break

        # compute combined slope and its variance
        if interact_name is not None and interact_name in params.index:
            try:
                beta_int = float(params[interact_name])
            except Exception:
                beta_int = float(np.asarray(params.loc[interact_name]))
            try:
                var_int = float(cov.loc[interact_name, interact_name]) if interact_name in cov.index else 0.0
            except Exception:
                var_int = 0.0
            cov_ai = cov_param(baseline_age_name, interact_name)
            combined_beta = beta_age + beta_int
            combined_var = var_age + var_int + 2.0 * cov_ai
        else:
            # no interaction term found -> slope equals baseline
            combined_beta = beta_age
            combined_var = var_age

        se = np.sqrt(combined_var) if (combined_var is not None and combined_var >= 0) else np.nan
        z = combined_beta / se if (not np.isnan(se) and se != 0) else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci = (combined_beta - 1.96 * se, combined_beta + 1.96 * se) if (not np.isnan(se)) else (np.nan, np.nan)

        results[label] = {
            "slope": combined_beta,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_95": ci,
            "interaction_param": interact_name  # None if no interaction term
        }

    # Attempt a joint Wald test of all interaction coefficients == 0
    joint_p = None
    try:
        if len(interaction_names) > 0:
            # Build restriction matrix R that tests each interaction coefficient = 0
            n_params = len(param_index)
            R = np.zeros((len(interaction_names), n_params))
            name_to_pos = {name: i for i, name in enumerate(param_index)}
            for r_i, iname in enumerate(interaction_names):
                pos = name_to_pos.get(iname, None)
                if pos is None:
                    # try alternate matching (e.g., different formatting)
                    matches = [i for i, nm in enumerate(param_index) if iname == nm or iname in nm or nm in iname]
                    pos = matches[0] if matches else None
                if pos is None:
                    raise KeyError(f"interaction param name {iname} not found in params index")
                R[r_i, pos] = 1.0
            # Use the model's wald_test method which should account for the result's covariance
            wres = glm_res.wald_test(R)
            # wres may have attribute .pvalue or .p_f
            joint_p = getattr(wres, "pvalue", None)
            if joint_p is None:
                joint_p = getattr(wres, "p_f", None)
            # ensure numeric
            if hasattr(joint_p, "__len__") and not isinstance(joint_p, (str, bytes)):
                joint_p = float(np.asarray(joint_p).ravel()[0])
            else:
                joint_p = float(joint_p)
    except Exception:
        joint_p = None  # leave as None if we cannot compute

    # Compose final object
    final_object = {
        "age_term_name": baseline_age_name,
        "per_culture_age_slopes": results,
        "interaction_terms": interaction_names,
        "interaction_joint_test_p": joint_p
    }

    # Prepare description: concise interpretation
    lines = []
    lines.append("Estimated age slopes for choosing the majority (positive => greater reliance on majority with age):")
    for cult in final_object["per_culture_age_slopes"].keys():
        r = final_object["per_culture_age_slopes"][cult]
        slope = r.get("slope", np.nan)
        p = r.get("p_value", np.nan)
        ci = r.get("ci_95", (np.nan, np.nan))
        se_val = r.get("se", np.nan)
        sig = ("p<0.05" if (p is not None and not (isinstance(p, float) and np.isnan(p)) and p < 0.05) else "ns")
        lines.append(f" - {cult}: slope={slope:.4f}, se={se_val:.4f}, 95%CI=({ci[0]:.4f}, {ci[1]:.4f}), {sig}")
    if final_object["interaction_terms"]:
        if joint_p is not None:
            lines.append(f"Joint test of age x culture interactions: p = {joint_p:.4g} "
                         f"({'evidence of differences' if joint_p < 0.05 else 'no strong evidence of differences'}).")
        else:
            lines.append("Could not compute a joint test for age x culture interactions; see per-culture p-values above.")
    else:
        lines.append("No age x culture interaction terms were present in the GLM (no evidence in model specification that slopes differ).")

    out["object"] = final_object
    out["description"] = "\n".join(lines)
    return out