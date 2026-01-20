def extract_final_answer(model_output):
    """
    Extract statistics relevant to how reliance on majority preference changes with age
    and whether that developmental trajectory differs across Sites (Age x Site interactions).

    Input:
      model_output : dict-like with keys 'glm_fit' and/or 'glm_robust' (statsmodels GLMResultsWrapper).
                     We prefer 'glm_robust' (fitted with cov_type='cluster') when available.

    Returns:
      dict with keys:
        - "object": dict containing:
            - coef_table: pandas.DataFrame with rows for all model parameters and columns
              ['coef', 'robust_se', 'z', 'pvalue', 'ci_lower', 'ci_upper'] (95% CI using robust SE)
            - age_terms: a sub-DataFrame filtered to rows for Age_c, Age_c2, and Age_c:Site interactions
            - interaction_names: list of interaction parameter names (Age_c x Site)
            - interaction_wald_test: dict with keys {'chi2', 'df', 'pvalue'} for joint test that all
              Age_c x Site interactions are zero (i.e., no difference in age slope across sites).
        - "description": brief plain-English interpretation of what the extracted objects mean.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Prefer cluster-robust fit if provided
    glm = None
    if isinstance(model_output, dict):
        glm = model_output.get('glm_robust') or model_output.get('glm_fit')
    else:
        glm = model_output

    if glm is None:
        raise ValueError("No GLM results found in model_output. Expected keys 'glm_robust' or 'glm_fit'.")

    # Extract parameter estimates and (robust) covariance matrix
    params = glm.params.copy()
    try:
        cov = glm.cov_params()
    except Exception:
        # fallback to result.cov_params_default if available, else use model-based cov
        try:
            cov = glm.cov_params_default
        except Exception:
            cov = glm.cov_params()  # will raise if nothing available

    param_names = params.index.tolist()
    cov = pd.DataFrame(cov, index=param_names, columns=param_names)

    # Standard errors, z-scores, p-values, CIs using normal approximation
    se = cov.values.diagonal() ** 0.5
    z_vals = params.values / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
    ci_lower = params.values - 1.96 * se
    ci_upper = params.values + 1.96 * se

    coef_table = pd.DataFrame({
        'coef': params.values,
        'robust_se': se,
        'z': z_vals,
        'pvalue': p_vals,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }, index=param_names)

    # Identify age-related terms:
    # - linear age main effect: "Age_c"
    # - quadratic age: "Age_c2"
    # - interactions: parameters containing "Age_c:" (Age_c x Site interactions)
    interaction_mask = [('Age_c:' in name) for name in param_names]
    age_c_mask = [name == 'Age_c' for name in param_names]
    age_c2_mask = [name == 'Age_c2' for name in param_names]

    interaction_names = [name for name, m in zip(param_names, interaction_mask) if m]
    age_related_names = [name for name in param_names if (name == 'Age_c' or name == 'Age_c2' or ('Age_c:' in name))]

    age_terms = coef_table.loc[age_related_names].copy()

    # Joint Wald test that all Age_c x Site interaction coefficients are zero
    # If there are no interaction terms, set test to None
    interaction_wald = None
    if len(interaction_names) > 0:
        # Build R matrix: rows = number interactions, cols = nparams. Each row selects one parameter.
        R = np.zeros((len(interaction_names), len(param_names)), dtype=float)
        for i, name in enumerate(interaction_names):
            j = param_names.index(name)
            R[i, j] = 1.0
        # Perform Wald test using the fitted result's covariance (this should respect the cov_type used when fitting)
        try:
            wald_res = glm.wald_test(R)
            # wald_res has attributes: statistic, df_denom, df_num, pvalue (or something similar)
            # Different statsmodels versions may expose different attribute names. We try common ones.
            stat = getattr(wald_res, 'statistic', None)
            pval = getattr(wald_res, 'pvalue', None)
            df_num = getattr(wald_res, 'df_denom', None) or getattr(wald_res, 'df_num', None)
            # If statistic is an array (e.g., chi2), try to extract scalar
            if isinstance(stat, (list, tuple, np.ndarray)):
                try:
                    stat_val = float(np.atleast_1d(stat).ravel()[0])
                except Exception:
                    stat_val = stat
            else:
                stat_val = stat
            interaction_wald = {
                'chi2': stat_val,
                'df': int(len(interaction_names)),
                'pvalue': float(pval) if pval is not None else None
            }
        except Exception:
            # If wald_test fails, compute a manual Wald chi-square:
            try:
                # beta vector for interactions
                b = params.loc[interaction_names].values
                cov_sub = cov.loc[interaction_names, interaction_names].values
                # Ensure invertible
                inv_cov = np.linalg.pinv(cov_sub)
                chi2 = float(b.T.dot(inv_cov).dot(b))
                pval = 1 - stats.chi2.cdf(chi2, df=len(interaction_names))
                interaction_wald = {'chi2': chi2, 'df': len(interaction_names), 'pvalue': float(pval)}
            except Exception:
                interaction_wald = {'chi2': None, 'df': len(interaction_names), 'pvalue': None}
    else:
        interaction_wald = {'chi2': None, 'df': 0, 'pvalue': None}

    result_object = {
        'coef_table': coef_table,
        'age_terms': age_terms,
        'interaction_names': interaction_names,
        'interaction_wald_test': interaction_wald
    }

    # Short description explaining contents and interpretation
    description_lines = []
    description_lines.append("Returned objects:")
    description_lines.append("- coef_table: coefficients, robust SEs, z, p-values, and 95% CIs for all model parameters.")
    description_lines.append("- age_terms: subset of coef_table for Age_c (linear age), Age_c2 (quadratic), and Age_c x Site interactions.")
    description_lines.append("- interaction_names: list of parameter names corresponding to Age_c x Site interaction terms.")
    description_lines.append("- interaction_wald_test: joint Wald chi-square test (and p-value) for the null that all Age_c x Site interaction coefficients = 0.")
    description_lines.append("")
    description_lines.append("How to interpret:")
    description_lines.append("- If Age_c has a significant coefficient (p < 0.05), there is evidence that reliance on the majority changes with age on average.")
    description_lines.append("- If Age_c2 is significant, the age trend is nonlinear (quadratic).")
    description_lines.append("- If the interaction_wald_test p-value < 0.05, developmental slopes (Age effect) differ across Sites (i.e., culture moderates the age trajectory).")
    description_lines.append("- You can inspect age_terms to see which specific site interactions are significant and their directions.")

    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }