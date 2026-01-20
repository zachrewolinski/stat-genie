def extract_final_answer(model_output):
    """
    Extract culture-specific age trajectories (linear and quadratic) for the
    probability of choosing the majority option from a fitted statsmodels MNLogit result.
    
    Returns a dict with:
      - "object": pandas.DataFrame with one row per culture (including a 'reference' culture)
                  and columns for linear & quadratic coefficients, standard errors, z-statistics,
                  and p-values for the majority-choice equation (log-odds of majority vs model reference).
      - "description": text describing what the numbers mean and important caveats.
    
    Notes / caveats:
      - The dependent coding assumed by the original model was: 0=unchosen, 1=majority, 2=minority.
        This function expects the majority category to be present among the modeled outcomes. If the
        majority category was used as the reference outcome when the model was fit, its coefficients
        will not be directly available and the function will return a descriptive message.
      - Combined standard errors for culture-specific coefficients are approximated by assuming
        independence (se_combined = sqrt(se_base^2 + se_inter^2)). If the full covariance matrix
        layout matches simple expectations, the code attempts but does not rely on using cov_params()
        to compute exact combined SEs. If exact covariances are needed, re-fitting or extracting
        covariances in the appropriate flattened format may be required.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic safety checks
    if not hasattr(res, 'params'):
        return {
            "object": None,
            "description": "Provided model_output has no .params attribute. Expected a statsmodels MNLogit result."
        }

    # Extract parameter tables (params, bse, pvalues, tvalues if present)
    params = res.params
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    tvalues = getattr(res, 'tvalues', None)

    # Ensure params is a DataFrame
    if isinstance(params, pd.Series):
        params = params.to_frame()
    params_df = params.copy()

    # Standard errors/pvalues may be DataFrame or Series; ensure DataFrames aligned with params
    if isinstance(bse, pd.Series):
        bse = bse.to_frame()
    if isinstance(pvalues, pd.Series):
        pvalues = pvalues.to_frame()
    if isinstance(tvalues, pd.Series):
        tvalues = tvalues.to_frame()

    # Convert column labels to string for robust matching
    params_df.columns = params_df.columns.astype(str)
    if bse is not None:
        try:
            bse.columns = bse.columns.astype(str)
        except Exception:
            pass
    if pvalues is not None:
        try:
            pvalues.columns = pvalues.columns.astype(str)
        except Exception:
            pass
    if tvalues is not None:
        try:
            tvalues.columns = tvalues.columns.astype(str)
        except Exception:
            pass

    # The majority category label (per problem statement) is "1"
    majority_label = '1'

    # Check whether majority is among modeled outcomes (i.e., has a column in params)
    if majority_label not in params_df.columns:
        # If not present, majority may have been used as reference in the MNLogit fit.
        return {
            "object": None,
            "description": (
                "The fitted multinomial model does not contain coefficients for the majority category (label '1'). "
                "This means the majority option was likely used as the reference (baseline) outcome when the model "
                "was estimated, so coefficients for 'majority' are not directly available. "
                "To obtain the developmental trajectories for choosing the majority option, re-fit the MNLogit model "
                "with a different reference category (e.g., set the minority or unchosen option as reference), "
                "or compute contrasts from the full covariance matrix after reparameterization."
            )
        }

    # Helper to safely extract scalar entries from param/bse/pval DataFrames
    def safe_get(df, row, col):
        if df is None:
            return np.nan
        try:
            return df.loc[row, col]
        except Exception:
            # Sometimes row labels may have unexpected whitespace or dtype; try fuzzy match
            if row in df.index:
                return df.loc[row, col] if col in df.columns else np.nan
            matches = [r for r in df.index if str(r).strip() == str(row).strip()]
            if matches:
                r = matches[0]
                return df.loc[r, col] if col in df.columns else np.nan
            return np.nan

    # Base (reference culture) coefficients for age terms in the majority equation
    base_age_coef = safe_get(params_df, 'age_c', majority_label)
    base_age_se = safe_get(bse, 'age_c', majority_label) if bse is not None else np.nan
    base_age_p = safe_get(pvalues, 'age_c', majority_label) if pvalues is not None else np.nan

    base_age2_coef = safe_get(params_df, 'age_c2', majority_label)
    base_age2_se = safe_get(bse, 'age_c2', majority_label) if bse is not None else np.nan
    base_age2_p = safe_get(pvalues, 'age_c2', majority_label) if pvalues is not None else np.nan

    # Identify interaction terms for cultures: those ending with '_x_age' and '_x_age2'
    all_param_names = list(params_df.index.astype(str))

    lin_int_names = [n for n in all_param_names if n.endswith('_x_age') and not n.endswith('_x_age2')]
    quad_int_names = [n for n in all_param_names if n.endswith('_x_age2')]

    # Derive culture names from interaction param names. Ex: 'culture_SiteB_x_age' -> 'culture_SiteB'
    def culture_from_interaction(param_name, suffix):
        return param_name[:-len(suffix)]

    cultures = []
    # baseline / reference culture
    cultures.append('reference')  # corresponds to the dropped culture dummy in the original design

    # map from culture label to its linear/quadratic interaction param names (or None)
    culture_map = {'reference': {'lin_name': None, 'quad_name': None}}

    # Collect all cultures from interactions
    found_cultures = []
    for lin in lin_int_names:
        cult = culture_from_interaction(lin, '_x_age')
        found_cultures.append(cult)
        culture_map[cult] = culture_map.get(cult, {})
        culture_map[cult]['lin_name'] = lin
    for quad in quad_int_names:
        cult = culture_from_interaction(quad, '_x_age2')
        found_cultures.append(cult)
        culture_map[cult] = culture_map.get(cult, {})
        culture_map[cult]['quad_name'] = quad

    # Add these cultures (sorted) to the cultures list
    unique_cultures = sorted(set(found_cultures))
    for c in unique_cultures:
        if c not in cultures:
            cultures.append(c)
    # Build results
    rows = []
    for cult in cultures:
        lin_name = culture_map.get(cult, {}).get('lin_name', None)
        quad_name = culture_map.get(cult, {}).get('quad_name', None)

        if cult == 'reference':
            lin_coef = base_age_coef
            lin_se = base_age_se
            lin_p = base_age_p
            quad_coef = base_age2_coef
            quad_se = base_age2_se
            quad_p = base_age2_p
        else:
            inter_lin_coef = safe_get(params_df, lin_name, majority_label) if lin_name is not None else 0.0
            inter_lin_se = safe_get(bse, lin_name, majority_label) if (bse is not None and lin_name is not None) else np.nan
            inter_lin_p = safe_get(pvalues, lin_name, majority_label) if pvalues is not None and lin_name is not None else np.nan

            inter_quad_coef = safe_get(params_df, quad_name, majority_label) if quad_name is not None else 0.0
            inter_quad_se = safe_get(bse, quad_name, majority_label) if (bse is not None and quad_name is not None) else np.nan
            inter_quad_p = safe_get(pvalues, quad_name, majority_label) if pvalues is not None and quad_name is not None else np.nan

            # Combined coefficients (reference + interaction)
            lin_coef = (base_age_coef if not pd.isna(base_age_coef) else 0.0) + (inter_lin_coef if not pd.isna(inter_lin_coef) else 0.0)
            quad_coef = (base_age2_coef if not pd.isna(base_age2_coef) else 0.0) + (inter_quad_coef if not pd.isna(inter_quad_coef) else 0.0)

            # Try to compute combined SE using covariance if possible; otherwise approximate by root-sum-squares
            lin_se_comb = np.nan
            quad_se_comb = np.nan
            try:
                cov = res.cov_params()
                # cov may be DataFrame with simple index or MultiIndex. We'll attempt several lookup strategies.
                def try_cov(a, b):
                    # a, b are param names (strings), both for the same outcome column majority_label.
                    # Possible formats of cov's index: simple param names (strings), or MultiIndex (param, outcome).
                    # Try simple lookup first.
                    if isinstance(cov.index, pd.MultiIndex):
                        # Try to find matching index keys where the outcome level equals majority_label
                        matches_a = [idx for idx in cov.index if str(idx[0]) == str(a) and str(idx[1]) == str(majority_label)]
                        matches_b = [idx for idx in cov.columns if str(idx[0]) == str(b) and str(idx[1]) == str(majority_label)]
                        if matches_a and matches_b:
                            return cov.loc[matches_a[0], matches_b[0]]
                        # Some versions use reversed level order; try matching by any element equal to a/b
                        matches_a = [idx for idx in cov.index if str(idx[0]) == str(a) or str(idx[1]) == str(a)]
                        matches_b = [idx for idx in cov.columns if str(idx[0]) == str(b) or str(idx[1]) == str(b)]
                        if matches_a and matches_b:
                            return cov.loc[matches_a[0], matches_b[0]]
                        # fallback: try flattening names
                        flat_index = ['_'.join([str(x) for x in idx]) for idx in cov.index]
                        flat_cols = ['_'.join([str(x) for x in idx]) for idx in cov.columns]
                        name_a = [i for i, s in enumerate(flat_index) if s.startswith(str(a)+'_') or s.endswith('_'+str(a)) or s==str(a)]
                        name_b = [j for j, s in enumerate(flat_cols) if s.startswith(str(b)+'_') or s.endswith('_'+str(b)) or s==str(b)]
                        if name_a and name_b:
                            return cov.iloc[name_a[0], name_b[0]]
                        raise KeyError
                    else:
                        # simple index
                        if (a in cov.index) and (b in cov.columns):
                            return cov.loc[a, b]
                        # try fuzzy matching
                        idx_a = [x for x in cov.index if str(x).strip() == str(a).strip()]
                        idx_b = [x for x in cov.columns if str(x).strip() == str(b).strip()]
                        if idx_a and idx_b:
                            return cov.loc[idx_a[0], idx_b[0]]
                        raise KeyError
                # For linear:
                if lin_name is not None:
                    try:
                        cov_aa = try_cov('age_c', 'age_c')
                        cov_bb = try_cov(lin_name, lin_name)
                        cov_ab = try_cov('age_c', lin_name)
                        lin_var = cov_aa + cov_bb + 2 * cov_ab
                        lin_se_comb = np.sqrt(max(lin_var, 0.0))
                    except Exception:
                        # fallback approximate
                        if not pd.isna(base_age_se) and not pd.isna(inter_lin_se):
                            lin_se_comb = np.sqrt(float(base_age_se)**2 + float(inter_lin_se)**2)
                        elif not pd.isna(base_age_se):
                            lin_se_comb = float(base_age_se)
                        else:
                            lin_se_comb = np.nan
                else:
                    lin_se_comb = base_age_se

                # For quadratic:
                if quad_name is not None:
                    try:
                        cov_aa = try_cov('age_c2', 'age_c2')
                        cov_bb = try_cov(quad_name, quad_name)
                        cov_ab = try_cov('age_c2', quad_name)
                        quad_var = cov_aa + cov_bb + 2 * cov_ab
                        quad_se_comb = np.sqrt(max(quad_var, 0.0))
                    except Exception:
                        if not pd.isna(base_age2_se) and not pd.isna(inter_quad_se):
                            quad_se_comb = np.sqrt(float(base_age2_se)**2 + float(inter_quad_se)**2)
                        elif not pd.isna(base_age2_se):
                            quad_se_comb = float(base_age2_se)
                        else:
                            quad_se_comb = np.nan
                else:
                    quad_se_comb = base_age2_se
            except Exception:
                # No covariance accessible or error: approximate
                if cult == 'reference':
                    lin_se_comb = base_age_se
                    quad_se_comb = base_age2_se
                else:
                    if not pd.isna(base_age_se) and not pd.isna(inter_lin_se):
                        lin_se_comb = np.sqrt(float(base_age_se)**2 + float(inter_lin_se)**2)
                    else:
                        lin_se_comb = base_age_se if not pd.isna(base_age_se) else inter_lin_se
                    if not pd.isna(base_age2_se) and not pd.isna(inter_quad_se):
                        quad_se_comb = np.sqrt(float(base_age2_se)**2 + float(inter_quad_se)**2)
                    else:
                        quad_se_comb = base_age2_se if not pd.isna(base_age2_se) else inter_quad_se

            lin_se = lin_se_comb
            quad_se = quad_se_comb

            # approximate p-values / z using normal approx
            lin_p = (2 * (1 - 0.5 * (1 + np.math.erf(abs(lin_coef / lin_se) / np.sqrt(2))))) if (not pd.isna(lin_se) and lin_se != 0) else np.nan
            quad_p = (2 * (1 - 0.5 * (1 + np.math.erf(abs(quad_coef / quad_se) / np.sqrt(2))))) if (not pd.isna(quad_se) and quad_se != 0) else np.nan

        lin_z = lin_coef / lin_se if (not pd.isna(lin_se) and lin_se != 0) else np.nan
        quad_z = quad_coef / quad_se if (not pd.isna(quad_se) and quad_se != 0) else np.nan

        rows.append({
            'culture': cult,
            'linear_coef_age_c': lin_coef,
            'linear_se': lin_se,
            'linear_z': lin_z,
            'linear_p': lin_p,
            'quad_coef_age_c2': quad_coef,
            'quad_se': quad_se,
            'quad_z': quad_z,
            'quad_p': quad_p
        })

    result_df = pd.DataFrame(rows)

    description = (
        "This table reports culture-specific estimated coefficients (and approximate SE/z/p) for the linear (age_c) "
        "and quadratic (age_c2) terms predicting the log-odds of choosing the majority option (category '1') relative to "
        "the model's reference outcome. A positive linear coefficient indicates that, for that culture, older children "
        "are more likely (higher log-odds) to choose the majority option; a negative coefficient indicates decreasing "
        "likelihood with age. The quadratic term captures curvature (e.g., acceleration or deceleration across age). "
        "Values for the 'reference' row correspond to the reference culture that was omitted when creating culture dummies. "
        "Other culture rows combine the reference coefficient with the culture-specific interaction (reference + interaction). "
        "Standard errors for combined coefficients were approximated; exact standard errors would require the full covariance "
        "matrix in the exact layout used by the fitted model. If the majority category was used as the model reference, the "
        "function returns a message and no coefficients (because coefficients for the reference outcome are not estimated directly)."
    )

    return {"object": result_df, "description": description}