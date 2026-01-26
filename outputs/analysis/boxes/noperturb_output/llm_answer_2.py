def extract_final_answer(model_output):
    """
    Extract statistics from a fitted statsmodels MNLogit results object to answer:
      How does children's reliance on majority preference develop with age across cultures?

    Returns a dict with:
      - "object": nested dict containing coefficients, SEs, z-stats, p-values, 95% CIs
                  for age effect on the majority choice (category 2 vs reference),
                  the age-by-culture interaction terms (for category 2), and estimated
                  age slopes by culture (with SE/conf if available).
      - "description": short textual interpretation of those statistics.

    Notes:
      - This function robustly handles different orientations of the results.params DataFrame
        returned by statsmodels MNLogit (params may be shaped with categories as rows or columns).
      - It attempts to compute standard errors for composite slopes (baseline + interaction)
        using the covariance matrix if parameter labels can be matched; otherwise SE is None.
    """
    import re
    import numpy as np
    import pandas as pd
    from math import isnan
    from scipy.stats import norm

    res = model_output  # statsmodels MNLogit results wrapper

    # Helper to safely get attributes
    params = getattr(res, "params", None)
    pvalues = getattr(res, "pvalues", None)
    bse = getattr(res, "bse", None)
    conf_int = None
    try:
        conf_int = res.conf_int()
    except Exception:
        conf_int = None

    # Convert params/pvalues/bse to DataFrame when possible.
    # params may be array-like or DataFrame/Series.
    try:
        params_df = pd.DataFrame(params)
    except Exception:
        params_df = None

    pval_df = None
    bse_df = None
    try:
        if pvalues is not None:
            pval_df = pd.DataFrame(pvalues)
    except Exception:
        pval_df = None
    try:
        if bse is not None:
            bse_df = pd.DataFrame(bse)
    except Exception:
        bse_df = None

    # Try to acquire exog names from the model object (robust fallback)
    exog_names_from_model = None
    try:
        if hasattr(res, "model") and hasattr(res.model, "exog_names"):
            exog_names_from_model = list(res.model.exog_names)
    except Exception:
        exog_names_from_model = None

    # If params_df exists but has numeric generic columns/indices, try to relabel using model exog names
    if params_df is not None and exog_names_from_model:
        # If number of columns matches number of exog names, set columns
        if params_df.shape[1] == len(exog_names_from_model):
            try:
                params_df.columns = exog_names_from_model
            except Exception:
                pass
        # If number of rows matches number of exog names, set index
        if params_df.shape[0] == len(exog_names_from_model):
            try:
                params_df.index = exog_names_from_model
            except Exception:
                pass

        # Also relabel pval_df and bse_df similarly if shapes match
        try:
            if pval_df is not None:
                if pval_df.shape[1] == len(exog_names_from_model):
                    pval_df.columns = exog_names_from_model
                if pval_df.shape[0] == len(exog_names_from_model):
                    pval_df.index = exog_names_from_model
        except Exception:
            pass
        try:
            if bse_df is not None:
                if bse_df.shape[1] == len(exog_names_from_model):
                    bse_df.columns = exog_names_from_model
                if bse_df.shape[0] == len(exog_names_from_model):
                    bse_df.index = exog_names_from_model
        except Exception:
            pass

    # Determine orientation: are exog names in columns or in index?
    orientation = None
    # Safe string lists for checking
    if params_df is None:
        # As last resort, try to construct params_df from numpy array with model shapes
        try:
            arr = np.asarray(params)
            if arr.ndim == 2 and exog_names_from_model:
                # assume rows are categories, columns are exog
                params_df = pd.DataFrame(arr, columns=exog_names_from_model)
            elif arr.ndim == 1 and exog_names_from_model:
                # 1D: assume it's a vector of exog coefficients (single category)
                params_df = pd.DataFrame([arr], columns=exog_names_from_model)
            else:
                params_df = pd.DataFrame(arr)
        except Exception:
            params_df = pd.DataFrame(params)

    cols_str = [str(c) for c in params_df.columns]
    idx_str = [str(i) for i in params_df.index]

    if "age_c" in params_df.columns or "age_c" in cols_str:
        orientation = "cols_exog"   # rows = categories, cols = exog names
    elif "age_c" in params_df.index or "age_c" in idx_str:
        orientation = "rows_exog"   # rows = exog names, cols = categories
    else:
        # Try to find 'age_c' among model.exog_names as final fallback and relabel if needed
        if exog_names_from_model and "age_c" in exog_names_from_model:
            # If columns match number of exog, assume columns are exog
            if params_df.shape[1] == len(exog_names_from_model):
                try:
                    params_df.columns = exog_names_from_model
                    orientation = "cols_exog"
                except Exception:
                    pass
            # If rows match number of exog, assign index
            if orientation is None and params_df.shape[0] == len(exog_names_from_model):
                try:
                    params_df.index = exog_names_from_model
                    orientation = "rows_exog"
                except Exception:
                    pass
        # As a more permissive search, check any column/index string contains 'age_c'
        if orientation is None:
            found_in_cols = any("age_c" == str(c) or "age_c" in str(c) for c in params_df.columns)
            found_in_idx = any("age_c" == str(i) or "age_c" in str(i) for i in params_df.index)
            if found_in_cols:
                orientation = "cols_exog"
            elif found_in_idx:
                orientation = "rows_exog"

    if orientation is None:
        # Final tolerant behavior: don't raise; return a helpful empty result
        return {
            "object": {},
            "description": "Could not locate 'age_c' among model parameters or model.exog_names. "
                           "Ensure the fitted model uses a predictor named exactly 'age_c'."
        }

    # Determine label for the majority category (we want category 2).
    def find_category_label(df, target=2):
        # labels are rows if orientation cols_exog, else columns
        if orientation == "cols_exog":
            labels = list(df.index)
            labels2 = list(df.columns)
        else:
            labels = list(df.columns)
            labels2 = list(df.index)
        # try matching target
        for lab in labels:
            try:
                if (lab == target) or (str(lab) == str(target)):
                    return lab
            except Exception:
                continue
        for lab in labels2:
            try:
                if (lab == target) or (str(lab) == str(target)):
                    return lab
            except Exception:
                continue
        for lab in labels + labels2:
            if str(lab) == "2":
                return lab
        # if not found, try to pick a plausible non-reference category:
        # choose first label found in df.index if orientation cols_exog, else first column
        if orientation == "cols_exog" and len(df.index) > 0:
            return df.index[0]
        if orientation == "rows_exog" and len(df.columns) > 0:
            return df.columns[0]
        raise ValueError("Could not find label for category '2' in model result.")

    try:
        cat2 = find_category_label(params_df, 2)
    except Exception:
        # If cannot locate, return informative message rather than raising
        return {
            "object": {},
            "description": "Could not identify the label for category '2' in the model parameters. "
                           f"Parameters index/columns: {list(params_df.index)} vs {list(params_df.columns)}"
        }

    # Accessor helpers for a coefficient given varname and category label
    def get_coef(varname, cat_label):
        try:
            if orientation == "cols_exog":
                return params_df.loc[cat_label, varname]
            else:
                return params_df.loc[varname, cat_label]
        except Exception:
            try:
                if orientation == "cols_exog":
                    return params_df.loc[cat_label, str(varname)]
                else:
                    return params_df.loc[str(varname), cat_label]
            except Exception:
                # if params_df contains flattened parameters, try to find by searching matching column/index names
                for col in params_df.columns:
                    if str(col) == str(varname) or str(col).endswith(str(varname)):
                        try:
                            if orientation == "cols_exog":
                                return params_df.loc[cat_label, col]
                            else:
                                return params_df.loc[col, cat_label]
                        except Exception:
                            continue
                return np.nan

    def get_pval(varname, cat_label):
        if pval_df is None:
            return None
        try:
            if orientation == "cols_exog":
                return pval_df.loc[cat_label, varname]
            else:
                return pval_df.loc[varname, cat_label]
        except Exception:
            try:
                if orientation == "cols_exog":
                    return pval_df.loc[cat_label, str(varname)]
                else:
                    return pval_df.loc[str(varname), cat_label]
            except Exception:
                return None

    def get_bse(varname, cat_label):
        if bse_df is None:
            return None
        try:
            if orientation == "cols_exog":
                return bse_df.loc[cat_label, varname]
            else:
                return bse_df.loc[varname, cat_label]
        except Exception:
            try:
                if orientation == "cols_exog":
                    return bse_df.loc[cat_label, str(varname)]
                else:
                    return bse_df.loc[str(varname), cat_label]
            except Exception:
                return None

    def get_confint(varname, cat_label):
        try:
            ci = res.conf_int()
            ci_df = pd.DataFrame(ci)
            # Try to locate row corresponding to (cat_label, varname) or similar
            # If ci_df has MultiIndex
            if isinstance(ci_df.index, pd.MultiIndex):
                candidates = [
                    (cat_label, varname), (str(cat_label), varname),
                    (cat_label, str(varname)), (str(cat_label), str(varname)),
                    (varname, cat_label), (str(varname), cat_label),
                    (varname, str(cat_label)), (str(varname), str(cat_label))
                ]
                for cand in candidates:
                    if cand in ci_df.index:
                        row = ci_df.loc[cand]
                        try:
                            lower, upper = row[0], row[1]
                            return float(lower), float(upper)
                        except Exception:
                            continue
            else:
                # flat index: try common concatenations
                flat = [str(x) for x in ci_df.index]
                candidates = [
                    f"{cat_label}.{varname}", f"{varname}.{cat_label}",
                    f"{cat_label}_{varname}", f"{varname}_{cat_label}",
                    f"{cat_label}:{varname}", f"{varname}:{cat_label}",
                    f"{str(cat_label)}.{str(varname)}", f"{str(varname)}.{str(cat_label)}"
                ]
                for cand in candidates:
                    if cand in flat:
                        i = flat.index(cand)
                        row = ci_df.iloc[i]
                        try:
                            lower, upper = row[0], row[1]
                            return float(lower), float(upper)
                        except Exception:
                            continue
            # fallback to param +/- 1.96*bse
            b = get_bse(varname, cat_label)
            c = get_coef(varname, cat_label)
            if (c is not None) and (b is not None) and (not isnan(b)):
                return float(c - 1.96 * b), float(c + 1.96 * b)
        except Exception:
            pass
        # final fallback using bse if available
        b = get_bse(varname, cat_label)
        c = get_coef(varname, cat_label)
        if (c is not None) and (b is not None) and (not isnan(b)):
            return float(c - 1.96 * b), float(c + 1.96 * b)
        return (None, None)

    # Extract main age effect for category 2 (majority choice vs reference)
    age_coef = get_coef("age_c", cat2)
    age_se = get_bse("age_c", cat2)
    age_p = get_pval("age_c", cat2)
    age_ci = get_confint("age_c", cat2)

    # Build list of exog names available
    if orientation == "cols_exog":
        exog_names = list(params_df.columns)
    else:
        exog_names = list(params_df.index)

    # Regex to capture patterns and the culture label
    pattern = re.compile(
        r'(?:age_c[:*]C\(culture\)\[T\.?([^\]]+)\]|C\(culture\)\[T\.?([^\]]+)\][:*]age_c|age_c:C\(culture\)\[T\.?([^\]]+)\]|C\(culture\)\[T\.?([^\]]+)\]:age_c)'
    )
    interactions = {}
    for name in exog_names:
        m = pattern.search(str(name))
        if m:
            groups = m.groups()
            culture_name = next((g for g in groups if g), None)
            if culture_name is None:
                continue
            coef_i = get_coef(name, cat2)
            se_i = get_bse(name, cat2)
            pv_i = get_pval(name, cat2)
            ci_i = get_confint(name, cat2)
            interactions[culture_name] = {
                "param_name": name,
                "coef": float(coef_i) if (coef_i is not None and not isnan(coef_i)) else None,
                "se": float(se_i) if (se_i is not None and not isnan(se_i)) else None,
                "p_value": float(pv_i) if (pv_i is not None and not isnan(pv_i)) else None,
                "ci_95": (float(ci_i[0]) if ci_i and ci_i[0] is not None else None,
                          float(ci_i[1]) if ci_i and ci_i[1] is not None else None)
            }

    # Attempt to compute slopes by culture for category 2:
    slopes = {}
    ref_label = "reference (omitted) culture"
    slopes[ref_label] = {
        "slope": float(age_coef) if (age_coef is not None and not isnan(age_coef)) else None,
        "se": float(age_se) if (age_se is not None and not isnan(age_se)) else None,
        "p_value": float(age_p) if (age_p is not None and not isnan(age_p)) else None,
        "ci_95": (float(age_ci[0]) if age_ci and age_ci[0] is not None else None,
                  float(age_ci[1]) if age_ci and age_ci[1] is not None else None)
    }

    # Covariance matrix (if available)
    cov_df = None
    try:
        cov = res.cov_params()
        cov_df = pd.DataFrame(cov)
    except Exception:
        cov_df = None

    def find_cov_label(category, varname):
        if cov_df is None:
            return None
        idx = cov_df.index
        if isinstance(idx, pd.MultiIndex):
            for cand in [(category, varname), (str(category), varname), (category, str(varname)), (str(category), str(varname))]:
                if cand in idx:
                    return cand
            for cand in [(varname, category), (str(varname), category), (varname, str(category)), (str(varname), str(category))]:
                if cand in idx:
                    return cand
        else:
            flat = [str(x) for x in idx]
            candidates = [
                f"{category}.{varname}", f"{varname}.{category}",
                f"{category}_{varname}", f"{varname}_{category}",
                f"{category}:{varname}", f"{varname}:{category}",
                f"{category} {varname}", f"{varname} {category}",
                f"{str(category)}.{str(varname)}", f"{str(varname)}.{str(category)}"
            ]
            for cand in candidates:
                if cand in flat:
                    return idx[flat.index(cand)]
        return None

    for cult, info in interactions.items():
        inter_name = info["param_name"]
        inter_coef = info["coef"]
        inter_se = info["se"]
        inter_p = info["p_value"]
        if (age_coef is None or (isinstance(age_coef, float) and isnan(age_coef))) and (inter_coef is None):
            slope_val = None
        else:
            a = 0.0 if (age_coef is None or (isinstance(age_coef, float) and isnan(age_coef))) else float(age_coef)
            b = 0.0 if (inter_coef is None or (isinstance(inter_coef, float) and isnan(inter_coef))) else float(inter_coef)
            slope_val = a + b

        slope_se = None
        slope_ci = (None, None)
        slope_p = None

        if cov_df is not None:
            lab_age = find_cov_label(cat2, "age_c")
            lab_inter = find_cov_label(cat2, inter_name)
            if lab_age is not None and lab_inter is not None:
                try:
                    var_age = cov_df.loc[lab_age, lab_age]
                    var_inter = cov_df.loc[lab_inter, lab_inter]
                    cov_ai = cov_df.loc[lab_age, lab_inter]
                    var_sum = float(var_age + var_inter + 2.0 * cov_ai)
                    if var_sum >= 0:
                        slope_se = float(np.sqrt(var_sum))
                        if slope_se > 0 and slope_val is not None:
                            z = slope_val / slope_se
                            slope_p = float(2.0 * (1.0 - norm.cdf(abs(z))))
                            slope_ci = (float(slope_val - 1.96 * slope_se), float(slope_val + 1.96 * slope_se))
                except Exception:
                    slope_se = None
                    slope_p = None
            else:
                flat_idx = list(map(str, cov_df.index))
                possible_age_labels = [f"{cat2}.{'age_c'}", f"{cat2}_age_c", f"age_c.{cat2}", f"age_c_{cat2}"]
                possible_inter_labels = [f"{cat2}.{str(inter_name)}", f"{cat2}_{str(inter_name)}",
                                         f"{str(inter_name)}.{cat2}", f"{str(inter_name)}_{cat2}"]
                found_age = None
                found_inter = None
                for lbl in possible_age_labels:
                    for i, x in enumerate(flat_idx):
                        if x == lbl:
                            found_age = cov_df.index[i]
                            break
                    if found_age is not None:
                        break
                for lbl in possible_inter_labels:
                    for i, x in enumerate(flat_idx):
                        if x == lbl:
                            found_inter = cov_df.index[i]
                            break
                    if found_inter is not None:
                        break
                if found_age is not None and found_inter is not None:
                    try:
                        var_age = cov_df.loc[found_age, found_age]
                        var_inter = cov_df.loc[found_inter, found_inter]
                        cov_ai = cov_df.loc[found_age, found_inter]
                        var_sum = float(var_age + var_inter + 2.0 * cov_ai)
                        if var_sum >= 0:
                            slope_se = float(np.sqrt(var_sum))
                            if slope_se > 0 and slope_val is not None:
                                z = slope_val / slope_se
                                slope_p = float(2.0 * (1.0 - norm.cdf(abs(z))))
                                slope_ci = (float(slope_val - 1.96 * slope_se), float(slope_val + 1.96 * slope_se))
                    except Exception:
                        slope_se = None
                        slope_p = None

        if slope_se is None:
            try:
                var_age_approx = (get_bse("age_c", cat2) ** 2) if get_bse("age_c", cat2) is not None else None
                var_inter_approx = (inter_se ** 2) if inter_se is not None else None
                if var_age_approx is not None and var_inter_approx is not None:
                    approx_var = float(var_age_approx + var_inter_approx)
                    slope_se = float(np.sqrt(approx_var))
                    if slope_se > 0 and slope_val is not None:
                        z = slope_val / slope_se
                        slope_p = float(2.0 * (1.0 - norm.cdf(abs(z))))
                        slope_ci = (float(slope_val - 1.96 * slope_se), float(slope_val + 1.96 * slope_se))
            except Exception:
                slope_se = None

        slopes[cult] = {
            "slope": float(slope_val) if (slope_val is not None and not isnan(slope_val)) else None,
            "se": float(slope_se) if (slope_se is not None and not isnan(slope_se)) else None,
            "p_value": float(slope_p) if (slope_p is not None and not isnan(slope_p)) else None,
            "ci_95": (float(slope_ci[0]) if slope_ci and slope_ci[0] is not None else None,
                      float(slope_ci[1]) if slope_ci and slope_ci[1] is not None else None)
        }

    result_object = {
        "majority_choice_category": 2,
        "age_effect_for_category_2": {
            "coef": float(age_coef) if (age_coef is not None and not isnan(age_coef)) else None,
            "se": float(age_se) if (age_se is not None and not isnan(age_se)) else None,
            "p_value": float(age_p) if (age_p is not None and not isnan(age_p)) else None,
            "ci_95": (float(age_ci[0]) if age_ci and age_ci[0] is not None else None,
                      float(age_ci[1]) if age_ci and age_ci[1] is not None else None)
        },
        "age_by_culture_interactions_for_category_2": interactions,
        "estimated_age_slopes_by_culture_for_category_2": slopes
    }

    def sig_label(p):
        if p is None:
            return "p=?"
        try:
            p = float(p)
            if p < 0.001:
                return "p<0.001"
            else:
                return f"p={p:.3f}"
        except Exception:
            return f"p={p}"

    desc_lines = []
    ae = result_object["age_effect_for_category_2"]
    if ae["coef"] is None:
        desc_lines.append("Could not extract the baseline age coefficient for the majority choice (category 2).")
    else:
        sign = "increase" if ae["coef"] > 0 else ("decrease" if ae["coef"] < 0 else "no linear change")
        se_text = f"SE={ae['se']:.4f}" if ae['se'] is not None else "SE=?"
        desc_lines.append(
            f"For choosing the majority option (category 2 vs reference), the baseline age coefficient is "
            f"{ae['coef']:.4f} ({se_text}), {sig_label(ae['p_value'])}; this indicates a "
            f"{sign} in log-odds of choosing the majority with increasing age in the reference culture."
        )

    if len(interactions) == 0:
        desc_lines.append("No age-by-culture interaction terms involving 'age_c' were found in the model output (i.e., no non-reference culture dummies detected).")
    else:
        significant_inter = []
        for cult, info in interactions.items():
            p = info.get("p_value")
            if p is not None and p < 0.05:
                significant_inter.append((cult, info))
        if significant_inter:
            sig_names = ", ".join([c for c, _ in significant_inter])
            desc_lines.append(
                f"Age-by-culture interactions were found for cultures: {', '.join(interactions.keys())}. "
                f"Significant interactions (p<.05) observed for: {sig_names}, indicating developmental change differs in those cultures."
            )
        else:
            desc_lines.append(
                f"Age-by-culture interactions were found for cultures: {', '.join(interactions.keys())}. "
                "None of the interaction coefficients reach p<0.05, suggesting no strong evidence that the age-related change in majority reliance differs across cultures."
            )

    desc_lines.append(
        "Interpretation: a positive age slope means older children are more likely to choose the majority option (greater reliance on majority preference), "
        "a negative slope means reliance declines with age. See 'object' for numeric coefficients, SEs, p-values, and 95% CIs."
    )

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}